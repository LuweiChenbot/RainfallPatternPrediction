# Task 2 models: what each forecast is and how it is made

Every forecast in Task 2, from the simplest baseline to the random forest, answers the same question in the same form. For station $s$ at time $t$, it gives one probability for each of the 9 classes of the next-30-minute rain total $Y$ (see `FEATURES.md`, "Label"):

$$
\mathbf{p}_{s,t} = (p_0, \dots, p_8), \qquad p_k = P(\text{label}_{s,t} = k \mid \text{information at } t), \qquad \sum_k p_k = 1 .
$$

The forecasts differ only in what information they condition on and how the probabilities are estimated. Every forecast stores this vector in a column called `probability`, so `evaluate.py` scores all of them the same way. The threshold mapper (section 6) then turns any of them into $P(Y > \theta)$.

Code:
- baselines: `code/baselines.py`, run in `notebooks/04_baselines.ipynb`;
- logistic regression and random forest: `code/models.py`, run in `notebooks/07_training.ipynb`;
- thresholds: `code/mapper.py`;
- scores: `code/evaluate.py`.

## At a glance

| Forecast | Conditions on | What it learns | How it is fitted | Fitted on |
|---|---|---|---|---|
| Constant | nothing | 9 class shares | counting | training table, with row weights |
| Climatology | station, month, hour | one 9-class distribution per station × month × hour, shrunk towards the island-wide month × hour | counting and shrinkage; shrinkage strength chosen on 2023 | every labelled row of 2017–2022 |
| Persistence | class of the last 30 minutes' rain | a 10 × 9 transition table | counting | every labelled row of 2017–2022 |
| Logistic regression | the 41 features, transformed | 9 × (D + 1) coefficients | weighted maximum likelihood with an L2 penalty; penalty chosen on 2023 | training table (`RAIN_TRAIN`) |
| Random forest | the 41 raw features | B decision trees with a class distribution in every leaf | greedy splits on bootstrap samples; settings chosen on 2023 | training table (`RAIN_TRAIN`) |

## Notation

| Symbol | Meaning |
|---|---|
| $i$ | one row: a station $s$ at a slot $t$ |
| $y_i \in \{0, \dots, 8\}$ | its class; $Y_i$ its 30-minute total in mm |
| $\mathbf{x}_i$ | its feature vector (`FEATURES.md`) |
| $w_i$ | row weight: $10$ for the dry rows kept in the subsampled training table, $1$ otherwise (`FEATURES.md`, "Row weight") |
| $e_0, \dots, e_7$ | class edges $0, 0.2, 0.5, 1, 2, 5, 10, 20$ mm. Class $0$ is $Y = 0$, class $k = 1, \dots, 7$ is $e_{k-1} < Y \le e_k$, class $8$ is $Y > 20$ |
| $\mathbb{1}[\cdot]$ | $1$ if the condition holds, else $0$ |

Years:
- **Training, 2017–2022:** the data every forecast is fitted on.
- **Validation, 2023:** used for every choice, from settings to models.
- **Test, 2024:** scored once, at the end.

## 1. Constant forecast

**Formula.** One distribution for every row:

$$
p_k = \frac{\sum_i w_i\, \mathbb{1}[y_i = k]}{\sum_i w_i}.
$$

**How it is fitted.** Counting on the training table. The weights undo the subsampling, so these are the true 2017–2022 class frequencies.

**What it learned** (9 numbers):

$$
\mathbf{p} = (0.93986,\ 0.01908,\ 0.00744,\ 0.01071,\ 0.00758,\ 0.00725,\ 0.00420,\ 0.00280,\ 0.00108),
$$

so it forecasts $P(Y > 0) = 0.060$ for every station at every time.

**How it predicts.** It returns the same vector for every row.

**Why it is there.** It is the "no information" reference. It has no ranking skill: its AUC-PR equals the event rate. On 2023 it scores RPS 0.1977 and log loss 0.3689.

**Code:** notebook 04, "Constant forecast". It has no function in `baselines.py`.

> In one sentence: it learned how often each rain class occurs overall, and forecasts exactly that for every station at every time.

## 2. Climatology

**Formula.** For station $s$, the month $m$ and the hour $h$ of the prediction time (Singapore time), start from training-row counts:

$$
n_{s,m,h,k} = \#\{\text{training rows of station } s \text{ in month } m \text{ and hour } h \text{ with class } k\},
\qquad n_{s,m,h} = \sum_k n_{s,m,h,k}.
$$

The island-wide share of class $k$ in month $m$ and hour $h$ pools all stations:

$$
q_{m,h,k} = \frac{\sum_s n_{s,m,h,k}}{\sum_s n_{s,m,h}} .
$$

The forecast is the station's own frequency, pulled towards that island-wide frequency:

$$
c_{s,m,h,k} = \frac{n_{s,m,h,k} + \kappa\, q_{m,h,k}}{n_{s,m,h} + \kappa}.
$$

- **$\kappa$** is the shrinkage strength (called `m` in the code).
  - $\kappa = 0$ gives the station's raw frequencies.
  - As $\kappa \to \infty$, every station gets the island-wide frequency for that month and hour.
- **Reading it as a prior.** $c$ is the average estimate you get by treating the island-wide shares as $\kappa$ imaginary extra rows for each station, then adding the station's real rows. (Formally, the posterior mean under a Dirichlet prior centred on $q_{m,h}$ with strength $\kappa$.)

**How it is fitted.**
- Counting on every labelled row of 2017–2022, with no subsampling and all weights 1.
- $\kappa$ is chosen from $\{0, 100, 1000, 3000, 10000\}$ by RPS on 2023. The validation RPS was 0.193995, 0.193886, 0.193572, **0.193505** and 0.193523, so **$\kappa = 3000$**.
- A station with a complete record has about 2,200 rows per month × hour cell (6 years × about 30 days × 12 slots). With $\kappa = 3000$, its own history therefore gets at most about 42% of the weight. Strong smoothing wins because, at the 30-minute scale, a station's month × hour history is mostly noise.

**What it learned.** Up to $87 \times 12 \times 24 = 25{,}056$ station × month × hour distributions (9 numbers each), plus $12 \times 24$ island-wide distributions.

**How it predicts.** It looks up $c_{s,\,\text{month}(t),\,\text{hour}(t)}$. If the station has no training rows in that month and hour, for example a station that started after 2022, it uses the island-wide $q_{\text{month}(t),\,\text{hour}(t)}$ instead.

**Why it is weak.** It knows nothing about the current weather, so it cannot tell a stormy afternoon from a dry one. On 2023 it scores RPS 0.1935, barely better than the constant forecast: time of day and season carry little information at the 30-minute scale.

**Code:** `fit_climatology`, `predict_climatology` in `baselines.py`.

> In one sentence: it learned how often each rain class occurs at each station in each month and hour of the day, strongly smoothed towards the island-wide pattern for that month and hour.

## 3. Persistence

**Formula.** Take the rain of the last 30 minutes (the current reading and the five before it):

$$
\text{past30}_{s,t} = \sum_{k=0}^{5} r_{s,t-k}.
$$

Its class $c$ uses the same edges as the label. If any of the six readings is missing, $c = -1$. The forecast is the distribution of the next 30 minutes' class given $c$:

$$
T_{c,k} = \frac{\sum_i w_i\, \mathbb{1}[c_i = c,\ y_i = k]}{\sum_i w_i\, \mathbb{1}[c_i = c]},
\qquad c \in \{-1, 0, \dots, 8\},\ k \in \{0, \dots, 8\}.
$$

This is the transition matrix of a first-order Markov chain on 30-minute rain classes: where the rain goes next, given where it is now.

**It is not "the next 30 minutes repeat the last 30".** That version would give class $c$ again with probability 1. This one gives the full distribution that followed class $c$ in the training years. So it is probabilistic, matches the training-year frequencies (it is calibrated), and spreads probability over every class that ever followed $c$.

**How it is fitted.** Counting on every labelled row of 2017–2022 (all weights 1). There is nothing to tune and no smoothing: every row of the table has thousands of training rows behind it.

**What it learned** (10 × 9 = 90 numbers; each row sums to 1). The column that matters most is the chance of any rain in the next 30 minutes, $1 - T_{c,0}$:

| Class $c$ of the last 30 minutes | Rain in the last 30 minutes | $P(\text{rain in the next 30 minutes})$ |
|---|---|---|
| $-1$ | unknown (a reading is missing) | 0.060 |
| $0$ | none | 0.024 |
| $1$ | $(0, 0.2]$ mm | 0.346 |
| $2$ | $(0.2, 0.5]$ mm | 0.591 |
| $3$ | $(0.5, 1]$ mm | 0.708 |
| $4$ | $(1, 2]$ mm | 0.790 |
| $5$ | $(2, 5]$ mm | 0.807 |
| $6$ | $(5, 10]$ mm | 0.868 |
| $7$ | $(10, 20]$ mm | 0.933 |
| $8$ | more than 20 mm | 0.980 |

**How it predicts.** It computes $c$ from the last 30 minutes and returns row $c$ of $T$.

**Strength and blind spot.**
- **Strength:** it is the strongest baseline (2023 RPS 0.1266), because rain persists and heavier rain persists more.
- **Blind spot:** every dry half hour gets the same 2.4%, so it cannot tell which dry stations are about to get rain. Its AUC-PR on the dry-now rows equals the event rate. These rain onsets are what the neighbour (Tier 2) features add.

**Code:** `now_class`, `fit_persistence`, `predict_persistence` in `baselines.py`.

> In one sentence: from six years of data it learned the probability distribution of the next half hour's rain class given the current half hour's rain class. After a dry half hour there is a 2.4% chance of rain; after more than 20 mm, 98%.

## 4. Multinomial logistic regression

**Inputs.** `build_lr` turns the features into a vector $\tilde{\mathbf{x}} \in \mathbb{R}^D$ in four steps (details in `FEATURES.md`, "Model-specific transformations"):

1. Apply $\log(1 + z)$ to rain amounts and the minutes since rain, and $\operatorname{sign}(z)\log(1 + |z|)$ to the two trend features.
2. One-hot encode `t0_season` into 2 columns, with the third season as the baseline.
3. Optionally (`nonlinear=True`), add the main features cut into one-hot bands, and the hour × season products.
4. Divide every column by its standard deviation in the training table (no centring).

This gives $D = 42$ inputs without the non-linear additions and $D = 137$ with them.

**Model.** The softmax form:

$$
z_k(\tilde{\mathbf{x}}) = \beta_{k,0} + \boldsymbol{\beta}_k^{\top}\tilde{\mathbf{x}},
\qquad
p_k(\tilde{\mathbf{x}}) = \frac{\exp z_k(\tilde{\mathbf{x}})}{\sum_{j=0}^{8} \exp z_j(\tilde{\mathbf{x}})}, \qquad k = 0, \dots, 8 .
$$

**How to read it.** The log-odds of class $k$ against "no rain" is a weighted sum of the inputs:

$$
\log\frac{p_k}{p_0} = (\beta_{k,0} - \beta_{0,0}) + (\boldsymbol{\beta}_k - \boldsymbol{\beta}_0)^{\top}\tilde{\mathbf{x}} .
$$

A one-standard-deviation increase in input $d$ multiplies the odds of class $k$ against class $0$ by $\exp(\beta_{k,d} - \beta_{0,d})$. Each input has one fixed effect per class, so the model needs the log scale and the bands to represent "a little rain matters a lot, more rain matters less".

**How it is trained.** By weighted maximum likelihood with an L2 penalty on the slopes (the intercepts are not penalised):

$$
\min_{\boldsymbol\beta}\ \ \frac{1}{\sum_i w_i} \sum_i w_i \big[-\log p_{y_i}(\tilde{\mathbf{x}}_i)\big] \;+\; \frac{\lambda}{2}\sum_{k=0}^{8} \lVert \boldsymbol{\beta}_k \rVert^2 .
$$

- **The first term is the log loss.** The weights $w_i$ put the dry subsampled rows back at their true frequency, so the fitted probabilities are on the real scale.
- **The solver.** Spark MLlib's L-BFGS runs until the objective changes by less than $10^{-6}$ (relative) or 1,000 iterations. In tests it converged after 435–633 iterations.
- **Why 9 coefficient vectors.** Adding the same vector to every class leaves the softmax unchanged, so 8 vectors would suffice. Spark keeps all 9, and the penalty selects the unique solution.

**What it learned.** An intercept vector (9 numbers) and a $9 \times D$ coefficient matrix: $9 \times 43 = 387$ parameters for the plain model, or $9 \times 138 = 1{,}242$ with the non-linear inputs. It also stores each column's standard deviation, estimated on the training table only.

**Tuning on 2023, by RPS.** One plain fit with $\lambda = 10^{-5}$, then the non-linear version with $\lambda \in \{10^{-6}, 10^{-5}, 10^{-4}\}$. The lowest validation RPS gives `lr_spec`; the table is in `results/lr_tuning.csv`.

**How it predicts.** Transform $\mathbf{x}$, take one matrix–vector product, apply the softmax, and return the 9 probabilities. The `prediction` column (the most likely class) is not used.

**Inspect it:** `PipelineModel.load("models/lr").stages[-1].coefficientMatrix` and `.interceptVector`.

**Code:** `build_lr` in `models.py`.

> In one sentence: it learned one weight per rain class and input, so that the log-odds of each rain class against no rain is a weighted sum of the station's recent rain, the neighbours' rain and the calendar terms. It is fitted by maximum likelihood with a small L2 penalty.

## 5. Random forest

**Inputs.** The 41 features as they are, with no transformations. A tree only compares a feature with thresholds, so a log or a rescaling changes nothing. `t0_season` enters as its code 0, 1 or 2.

**One tree.** A classification tree, grown as follows:

- **Data.** A bootstrap sample of the training table, Spark's default. Each row enters tree $b$ a random number of times $c_{ib} \sim \text{Poisson}(\rho)$, with $\rho = 0.5$ (`subsampling`). That means about 39% of rows, some repeated. In tree $b$, row $i$ counts with weight $w_i c_{ib}$.
- **Candidate thresholds.** For each feature, up to `maxBins − 1` thresholds at approximate quantiles of the training values.
- **Splits.** At each node, draw a random subset of the features: `sqrt` gives 7, `onethird` 14, `0.5` 21 of the 41. Among those, choose the feature $d$ and threshold $\tau$ that most reduce the weighted impurity:

$$
\Delta = I(\text{node}) - \frac{W_L}{W}\, I(\text{left}) - \frac{W_R}{W}\, I(\text{right}),
\qquad
I_{\text{Gini}} = 1 - \sum_k \pi_k^2,
\qquad
I_{\text{entropy}} = -\sum_k \pi_k \log_2 \pi_k,
$$

  where $\pi_k$ is the weighted share of class $k$ among the rows in the node, and $W, W_L, W_R$ are the total weights in the node and its two children.
- **Stopping.** A node is not split further at depth `maxDepth`, when either child would have fewer than `minInstancesPerNode` rows, or when no split reduces impurity.
- **Leaves.** Each leaf $\ell$ stores the weighted class distribution $\boldsymbol\pi^{(\ell)}$ of the training rows that reached it.

**The forest's forecast is a soft vote.** It averages, over the $B$ trees, the class distribution of the leaf that $\mathbf{x}$ falls into:

$$
p_k(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} \pi^{(\ell_b(\mathbf{x}))}_k ,
$$

where $\ell_b(\mathbf{x})$ is the leaf of tree $b$ that $\mathbf{x}$ reaches.

**How it is "trained".** There is no single loss being minimised:
- each split is chosen greedily for impurity;
- randomness (bootstrap rows and random feature subsets) makes the trees differ from each other;
- averaging many different trees reduces the variance of the forecast.

The row weights enter both the impurity and the leaf distributions, so, as with the logistic regression, the probabilities are on the true rain-frequency scale.

**What it learned.**
- $B$ trees. Each internal node holds a rule "feature $d \le \tau$", and each leaf a 9-class distribution.
- It finds thresholds and interactions on its own. One path might read "dry now, more than half of the stations within 10 km wet, afternoon".
- For scale: the first 50-tree, depth-14 forest had 226,340 nodes (8.4 MB).

**Settings, chosen on 2023 by RPS** (`results/rf_screening.csv`). Fixed: `subsampling` 0.5 and seed 42. Screened one at a time:
- depth: 14, 17, 20, 24;
- minimum rows per leaf: 20, 50, 100, 300;
- features per split: `sqrt`, `onethird`, `0.5`;
- `maxBins`: 64, 256;
- impurity: Gini, entropy.

The number of trees comes from `RAIN_RF_TREES`. `maxMemoryInMB` and `cacheNodeIds` affect speed only, not the result.

**Feature importance** (`featureImportances`). For each feature, the total weighted impurity decrease of all splits on it, normalised within each tree and averaged over trees. It favours continuous features, which offer many possible thresholds.

**Inspect it:** `.stages[-1].totalNumNodes`, `.featureImportances`, and `.trees[0].toDebugString` to print one tree's rules (long).

**Code:** `build_rf` in `models.py`.

> In one sentence: it learned many decision trees, each splitting the training rows by thresholds on the features into groups with similar next-half-hour rain. A forecast averages the rain-class distributions of the groups the new row lands in, one group per tree.

## 6. From class probabilities to $P(Y > \theta)$: the threshold mapper

Every forecast above gives 9 class probabilities, but the task asks for $P(Y > \theta)$ at any threshold $\theta$.

For $\theta \ge 0$, let $k$ be the class that contains $\theta$, i.e. $e_{k-1} \le \theta < e_k$ (with $e_8 = \infty$). Then

$$
P(Y > \theta) = \sum_{j > k} p_j \;+\; p_k \, q_k(\theta),
\qquad
q_k(\theta) = \frac{\sum_{i:\ y_i = k,\ Y_i > \theta} w_i}{\sum_{i:\ y_i = k} w_i}.
$$

- **Classes entirely above $\theta$ count fully.** The class containing $\theta$ counts with the share of its training rows that lie above $\theta$.
- **$q_k$ is learned once,** from the training table: the weighted list of $Y$ values inside each wet class, saved in `models/mapper.json`. The same mapper serves every model.
- **At a class edge** $\theta = e_j$, $q = 1$, and the formula reduces to $\sum_{k > j} p_k$. That is what `evaluate.add_exceedance` computes.
- **Assumption:** inside a class, $Y$ is distributed as in the training years. The model decides how likely each class is; the mapper only splits a class.
- **Limitation:** the largest training value in class 8 is 71.8 mm, so $P(Y > \theta) = 0$ for $\theta \ge 71.8$ mm.

**Code:** `fit`, `locate`, `exceedance`, `add_threshold` in `mapper.py`.

## 7. How the forecasts are compared

Every choice is made on 2023; 2024 is scored once, at the end. Validation and test rows have weight 1, so the weighted scores below are plain averages there.

- **Brier score** at edge $e_j$:

$$
\text{BS}_j = \frac{1}{\sum_i w_i}\sum_i w_i \big(P_i(Y > e_j) - \mathbb{1}[Y_i > e_j]\big)^2 .
$$

- **Ranked probability score**, the selection metric, sums the Brier scores over the 8 edges:

$$
\text{RPS} = \sum_{j=0}^{7} \text{BS}_j .
$$

  This is the squared error of the whole cumulative forecast, so near misses (probability on a neighbouring class) cost less than far misses. Lower is better; the constant forecast scores 0.1977.
- **Log loss:**

$$
-\frac{1}{\sum_i w_i}\sum_i w_i \log p_{y_i}(\mathbf{x}_i).
$$

- **AUC-PR** at $\theta$: how well $P(Y > \theta)$ ranks the rare rows with $Y > \theta$ above the rest. A forecast with no skill scores the event rate.
- **Reliability:** forecasts grouped into 10 equal-width probability bins, comparing each bin's mean forecast with the observed frequency (calibration).

**Training goal and selection metric differ.**
- The baselines are fitted by counting, which is maximum likelihood for their simple form.
- The logistic regression minimises weighted log loss plus the penalty.
- The random forest makes greedy impurity splits.
- Settings, and the comparison between models, use RPS on 2023.
