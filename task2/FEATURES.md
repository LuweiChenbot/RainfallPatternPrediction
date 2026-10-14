# Task 2 features: definitions

How the label and each model feature are computed, written as formulas: 41 features in Tiers 0 to 2, plus 17 in Tier 3. Every quantity is computed from information available at the prediction time $t$ or earlier. The only exceptions are the two station wetness ratios: they use labels, but only from the training years.

Code: `task2/code/features.py` (time, Tier 1, Tier 2, Tier 3), `task2/notebooks/01_labels.ipynb` (cleaning, label), `task2/notebooks/02_station_feature.ipynb` (station features), `task2/notebooks/08_tier3_features.ipynb` (Tier 3), `task2/notebooks/03_train_valid_test_split.ipynb` (joining and weights).

## Notation and data sources

| Symbol | Meaning | Source |
|---|---|---|
| $s$, $j$ | a station ($s$ the station being forecast, $j$ another station) | `task1_revision/data/matrix_5min/stations.csv`, 91 stations |
| $t$ | a five-minute slot, the prediction time; slot $0$ is 2017-01-01 00:00 Singapore time | `config.ORIGIN_EPOCH`, `config.SLOT_SECONDS` |
| $r_{s,t}$ | rainfall (mm) recorded by station $s$ at slot $t$, after cleaning (below) | `task1_revision/data/matrix_5min/`, built by Spark from the aligned observations |
| $a_{s,t}$ | $1$ if $r_{s,t}$ is available (not missing), else $0$ | |
| $(x_s, y_s)$ | station position in km east and north of the island centre | latest recorded longitude and latitude in `stations.csv` |
| $d_{sj}$ | distance between stations $s$ and $j$ (km) | from $(x, y)$ |
| $\theta_{sj}$ | bearing from $s$ to $j$ in degrees, $0$ = north, clockwise | from $(x, y)$ |
| $\mathbb{1}[\cdot]$ | indicator: $1$ if the condition holds, else $0$ | |

**Cleaning.** Before any label or feature is computed, two kinds of readings are set to missing ($a_{s,t} = 0$):

- **Stuck gauges.** These are station-months in which the gauge recorded no wet reading although, given how often the network was wet, at least 30 were expected. With $p_t$ the share of reporting stations that are wet at slot $t$, a station-month $m$ is stuck when

$$
O_{s,m} = \sum_{t \in m} a_{s,t}\,\mathbb{1}[r_{s,t} > 0] = 0
\quad\text{and}\quad
E_{s,m} = \sum_{t \in m} a_{s,t}\, p_t \ge 30,
\qquad
p_t = \frac{\sum_j a_{j,t}\,\mathbb{1}[r_{j,t} > 0]}{\sum_j a_{j,t}} .
$$

  This flags 18 station-months, listed in `task2/data/stuck_station_months.csv`.
- **Faulty periods.** These are readings of rain that never fell: S107 from 2022-08-19 22:00 to 2022-08-22 06:00, and S36 from 2019-02-01 to 2021-09-01. They are listed in `config.FAULTY`, with the S107 evidence in `notebooks/S107_gauge_fault_2022_08.txt`.

**Positions.** Each station is projected onto a flat km grid around $(\lambda_0, \varphi_0) = (103.82^\circ\text{E}, 1.35^\circ\text{N})$ with an equirectangular projection, which is accurate to well under 1% across Singapore:

$$
x_s = (\lambda_s - \lambda_0)\cdot 111.32 \cos\varphi_0, \qquad
y_s = (\varphi_s - \varphi_0)\cdot 110.57,
$$

$$
d_{sj} = \sqrt{(x_j - x_s)^2 + (y_j - y_s)^2}, \qquad
\theta_{sj} = \operatorname{atan2}(x_j - x_s,\; y_j - y_s) \bmod 360^\circ .
$$

## Label

The target is the rain total over the next 30 minutes, i.e. the next six readings:

$$
Y_{s,t} = \sum_{k=1}^{6} r_{s,t+k},
$$

rounded to 0.1 mm, because the gauge measures in 0.2 mm steps and the stored values carry float noise.

A row $(s, t)$ is kept only if all of these hold:
- the current reading exists ($a_{s,t} = 1$);
- all six future readings exist ($\sum_{k=1}^{6} a_{s,t+k} = 6$);
- the 30-minute window ends in the same calendar year as $t$, so no window crosses the train/validation/test boundary.

That leaves 47,511,766 rows.

The class label bins $Y$ at the edges $0, 0.2, 0.5, 1, 2, 5, 10, 20$ mm. Each split point sits 0.05 mm above its edge, so a value exactly on an edge (e.g. 0.2) stays in the lower class:

$$
\text{label}_{s,t} = k \quad\text{if}\quad b_k \le Y_{s,t} < b_{k+1},
\qquad (b_0, \dots, b_9) = (-\infty, 0.05, 0.25, 0.55, 1.05, 2.05, 5.05, 10.05, 20.05, \infty).
$$

The model predicts $P(\text{label} = k)$ for $k = 0, \dots, 8$. The threshold mapper (`code/mapper.py`) turns these class probabilities into $P(Y_{s,t} > \theta)$ for any threshold $\theta$.

Two columns are kept for evaluation only; neither is a feature:

$$
\text{past30\_mm}_{s,t} = \sum_{k=0}^{5} r_{s,t-k}, \qquad
\text{regime} = \text{wet\_now if past30\_mm} > 0,\ \text{dry\_now if} = 0,\ \text{unknown if a reading is missing}.
$$

## Tier 0: station and calendar (15 features)

### `t0_x_km`, `t0_y_km`: station position

$$
\texttt{t0\_x\_km} = x_s, \qquad \texttt{t0\_y\_km} = y_s .
$$

- $x_s, y_s$: as defined in the notation above.
- Source: `stations.csv`, computed in notebook 02.

### `t0_wet_ratio`, `t0_heavy_ratio`: how rainy the station is compared with the network

These compare the number of rainy windows the station had with the number expected if it behaved like the network average at the same times:

$$
\texttt{t0\_wet\_ratio}_s = \frac{O^{\text{wet}}_s + M}{E^{\text{wet}}_s + M},
\qquad
O^{\text{wet}}_s = \sum_{t \in \mathcal{T}_s} \mathbb{1}[Y_{s,t} > 0],
\qquad
E^{\text{wet}}_s = \sum_{t \in \mathcal{T}_s} \bar p^{\,\text{wet}}_t,
\qquad
\bar p^{\,\text{wet}}_t = \frac{1}{|\mathcal{S}_t|}\sum_{j \in \mathcal{S}_t} \mathbb{1}[Y_{j,t} > 0] .
$$

- $\mathcal{T}_s$: the slots of station $s$ that have a label, in the training years 2017–2022 only, so the validation and test years never influence the feature.
- $\mathcal{S}_t$: the stations with a label at slot $t$.
- $O$ ("observed") and $E$ ("expected"): a ratio above 1 means the station is wetter than the network at the same times. Comparing at the same times means a station that only reported in a dry year is not penalised.
- $M = 500$: shrinkage. Adding it to both counts pulls stations with short records towards 1.

`t0_heavy_ratio` is the same with $\mathbb{1}[\text{label} \ge 6]$, i.e. $Y > 5$ mm, in place of $\mathbb{1}[Y > 0]$. Both are constant per station; source `station_static.csv`, written by notebook 02.

### `t0_hod_s{k}`, `t0_hod_c{k}`, $k = 1, 2, 3$: time of day

$$
h_t = \text{hour}(t) + \frac{\text{minute}(t)}{60} \in [0, 24),
\qquad
\texttt{t0\_hod\_s}k = \sin\!\Big(\frac{2\pi k\, h_t}{24}\Big),
\quad
\texttt{t0\_hod\_c}k = \cos\!\Big(\frac{2\pi k\, h_t}{24}\Big).
$$

- $h_t$ is the prediction time in Singapore local time.
- The sine/cosine pair makes 23:55 and 00:00 neighbours. Three harmonics ($k = 1, 2, 3$) can reproduce the sharp afternoon peak of the daily cycle; in Task 1 they explain 82% of its variation.

### `t0_doy_s{k}`, `t0_doy_c{k}`, $k = 1, 2$: time of year

$$
\delta_t = \frac{\text{dayofyear}(t) - 1 + h_t/24}{N_{\text{year}(t)}} \in [0, 1),
\qquad
\texttt{t0\_doy\_s}k = \sin(2\pi k\, \delta_t),
\quad
\texttt{t0\_doy\_c}k = \cos(2\pi k\, \delta_t).
$$

- $N_{\text{year}}$: 365 or 366 days in the year.
- Two harmonics correspond to Task 1's finding that an annual wave plus a half-year wave capture the seasonal cycle.

### `t0_season`: monsoon season

$$
\texttt{t0\_season} =
\begin{cases}
0 & \text{month} \in \{12, 1, 2, 3\} \text{ (northeast monsoon)}\\
2 & \text{month} \in \{6, 7, 8, 9\} \text{ (southwest monsoon)}\\
1 & \text{otherwise (inter-monsoon)}
\end{cases}
$$

The random forest uses this code directly. The logistic regression one-hot encodes it.

## Tier 1: the station's own recent rain (11 features)

All Tier 1 features use readings at slots $t, t-1, \dots$ of the same station, so they include the current reading. Missing readings are skipped, never treated as zero. Let $\mathcal{W}_n(t) = \{t-n+1, \dots, t\}$ be the last $n$ slots.

### `t1_rain_5`, `t1_rain_15`, `t1_rain_30`, `t1_rain_60`, `t1_rain_120`, `t1_rain_1440`: rain over the past $L$ minutes

$$
\texttt{t1\_rain\_}L_{s,t} = \sum_{u \in \mathcal{W}_{L/5}(t)} a_{s,u}\, r_{s,u},
\qquad L \in \{5, 15, 30, 60, 120, 1440\},
$$

rounded to 0.1 mm. `t1_rain_5` is the current reading $r_{s,t}$ itself.

### `t1_max5_30`: strongest five-minute reading in the past 30 minutes

$$
\texttt{t1\_max5\_30}_{s,t} = \max_{u \in \mathcal{W}_6(t),\ a_{s,u}=1} r_{s,u} .
$$

This separates a short burst from steady light rain with the same total.

### `t1_wet_slots_60`: wet readings in the past hour

$$
\texttt{t1\_wet\_slots\_60}_{s,t} = \sum_{u \in \mathcal{W}_{12}(t)} a_{s,u}\,\mathbb{1}[r_{s,u} > 0] \in \{0, \dots, 12\}.
$$

### `t1_trend_30`: is the rain building or easing?

$$
\texttt{t1\_trend\_30}_{s,t} = 2\,R^{15}_{s,t} - R^{30}_{s,t} = R^{15}_{s,t} - \big(R^{30}_{s,t} - R^{15}_{s,t}\big),
$$

- $R^{15}, R^{30}$: `t1_rain_15` and `t1_rain_30`.
- The result is the last 15 minutes minus the 15 minutes before them: positive when the rain is intensifying.

### `t1_min_since_rain`: minutes since the last wet reading

$$
\texttt{t1\_min\_since\_rain}_{s,t} = \min\!\Big(1440,\; 5\,\big(t - \max\{u \le t : a_{s,u} = 1,\ r_{s,u} > 0\}\big)\Big),
$$

- It is 0 when it is raining now.
- It is capped at 1440 minutes (24 h), the value also used when the station has never recorded rain. In the validation year, 44% of rows sit at the cap.

### `t1_missing_120`: missing readings in the past two hours

$$
\texttt{t1\_missing\_120}_{s,t} = 24 - \sum_{u \in \mathcal{W}_{24}(t)} a_{s,u} .
$$

This tells the model when the rain sums above cover fewer readings than usual.

## Tier 2: rain at other stations (15 features)

Tier 2 uses the Tier 1 totals of the other stations at the same slot $t$:
- $R^{15}_{j,t}$ = `t1_rain_15`, $R^{30}_{j,t}$ = `t1_rain_30`;
- $a_{j,t}$ = whether station $j$ reported at $t$.

Station $s$ itself is always excluded ($d_{ss} = 0$), except in the two island-wide features.

**Weighted neighbour mean.** For a neighbour weight $w_{sj} \ge 0$ and a quantity $z_{j,t}$:

$$
\operatorname{mean}_w(z)_{s,t} =
\begin{cases}
\dfrac{\sum_j w_{sj}\, a_{j,t}\, z_{j,t}}{\sum_j w_{sj}\, a_{j,t}} & \text{if } \sum_j w_{sj}\, a_{j,t} > 0,\\[2ex]
0 & \text{otherwise (no reporting neighbour).}
\end{cases}
$$

Only stations that reported at $t$ count, so a missing neighbour neither adds rain nor dilutes the mean.

**Weights** (`tier2_weights` in `features.py`):

$$
w^{\text{idw}}_{sj} = \frac{\mathbb{1}[0 < d_{sj} \le 10]}{\max(d_{sj}, 0.5)^2},
\qquad
w^{\text{far}}_{sj} = \frac{\mathbb{1}[10 < d_{sj} \le 25]}{\max(d_{sj}, 0.5)^2},
$$

$$
w^{5\text{km}}_{sj} = \mathbb{1}[0 < d_{sj} \le 5],
\qquad
w^{10\text{km}}_{sj} = \mathbb{1}[0 < d_{sj} \le 10],
\qquad
w^{q}_{sj} = \mathbb{1}[0 < d_{sj} \le 25]\;\mathbb{1}\big[\lfloor ((\theta_{sj} + 45) \bmod 360) / 90 \rfloor = q\big].
$$

- **idw** (inverse distance weighting): nearer stations count more, with weight $1/d^2$. The $0.5$ km floor stops very close pairs from dominating.
- **Quadrants:** $q = 0, 1, 2, 3$ are north, east, south and west. Each covers $90^\circ$ centred on that compass direction; north is $315^\circ$–$45^\circ$.

### `t2_idw_rain_15`, `t2_idw_rain_30`: rain nearby (within 10 km)

$$
\texttt{t2\_idw\_rain\_15} = \operatorname{mean}_{w^{\text{idw}}}(R^{15}), \qquad
\texttt{t2\_idw\_rain\_30} = \operatorname{mean}_{w^{\text{idw}}}(R^{30}).
$$

### `t2_idw_trend`: is the rain nearby building or easing?

$$
\texttt{t2\_idw\_trend} = 2\,\texttt{t2\_idw\_rain\_15} - \texttt{t2\_idw\_rain\_30},
$$

which is the neighbours' version of `t1_trend_30`.

### `t2_far_rain_30`: rain 10–25 km away

$$
\texttt{t2\_far\_rain\_30} = \operatorname{mean}_{w^{\text{far}}}(R^{30}).
$$

This is rain that may still be on its way. Task 1 found that storms move at 38–59 km/h, i.e. 10–25 km in roughly 10–40 minutes.

### `t2_max_rain_30`: heaviest rain at any station within 10 km

$$
\texttt{t2\_max\_rain\_30}_{s,t} = \max\Big(0,\ \max_{j:\ 0 < d_{sj} \le 10,\ a_{j,t}=1} R^{30}_{j,t}\Big).
$$

### `t2_wet_frac_5km`, `t2_wet_frac_10km`: share of nearby stations where it has rained

$$
\texttt{t2\_wet\_frac\_5km} = \operatorname{mean}_{w^{5\text{km}}}\big(\mathbb{1}[R^{30} > 0]\big), \qquad
\texttt{t2\_wet\_frac\_10km} = \operatorname{mean}_{w^{10\text{km}}}\big(\mathbb{1}[R^{30} > 0]\big).
$$

Each is the fraction of reporting stations within 5 or 10 km that had any rain in the past 30 minutes.

### `t2_dist_rain_km`: distance to the nearest rain

$$
\texttt{t2\_dist\_rain\_km}_{s,t} = \min\Big(30,\ \min_{j \ne s:\ a_{j,t}=1,\ R^{15}_{j,t} > 0} d_{sj}\Big).
$$

- The nearest station with rain in the past 15 minutes, at any distance.
- Capped at 30 km. The cap also stands for "no station is raining", which is the case in 74% of validation rows.

### `t2_rain_30_n`, `t2_rain_30_e`, `t2_rain_30_s`, `t2_rain_30_w`: rain by direction (within 25 km)

$$
\texttt{t2\_rain\_30\_}q = \operatorname{mean}_{w^{q}}(R^{30}), \qquad q \in \{n, e, s, w\}.
$$

Each is the plain average over the reporting stations in that quadrant. These features let the model use storm direction: in the southwest monsoon, rain to the west arrives first.

### `t2_island_wet_frac`, `t2_island_rain_30`: the whole network

$$
\texttt{t2\_island\_wet\_frac}_t = \frac{\sum_j a_{j,t}\,\mathbb{1}[R^{30}_{j,t} > 0]}{\max\big(1, \sum_j a_{j,t}\big)},
\qquad
\texttt{t2\_island\_rain\_30}_t = \frac{\sum_j a_{j,t}\, R^{30}_{j,t}}{\max\big(1, \sum_j a_{j,t}\big)}.
$$

These average over all reporting stations, station $s$ included, so they are the same for every station at slot $t$. They describe whether it is a rainy day island-wide.

### `t2_nbrs_10km`: how many neighbours are reporting

$$
\texttt{t2\_nbrs\_10km}_{s,t} = \sum_j w^{10\text{km}}_{sj}\, a_{j,t}.
$$

This tells the model how many stations the 5 km and 10 km averages above are based on.

All Tier 2 values are rounded to 4 decimals.

## Tier 3: how the neighbourhood is changing, and rain upstream (17 features)

Code: `add_tier3`, `upstream_kernels`, `season_of` in `features.py`, run in `notebooks/08_tier3_features.ipynb`. The settings are in `config.py`: `LAG_SLOTS`, `STORM_MOTION`, `UPSTREAM_MINUTES`, `UP_SIGMA_KM`, `UP_RADIUS_KM`.

### `t3_d15_*`: change of a Tier 2 summary over the last 15 minutes (12 features)

Write $F_{s}\big(R^{15}_{\cdot,t}, R^{30}_{\cdot,t}, a_{\cdot,t}\big)$ for a Tier 2 summary of station $s$, computed from every station's readings at slot $t$. The change feature recomputes the same summary from the readings $\ell = 3$ slots (15 minutes) earlier and takes the difference:

$$
\texttt{t3\_d15\_}F_{s,t} = F_{s}\big(R^{15}_{\cdot,t}, R^{30}_{\cdot,t}, a_{\cdot,t}\big) - F_{s}\big(R^{15}_{\cdot,t-\ell}, R^{30}_{\cdot,t-\ell}, a_{\cdot,t-\ell}\big),
$$

for $F \in \{$`idw_rain_30`, `far_rain_30`, `max_rain_30`, `wet_frac_5km`, `wet_frac_10km`, `dist_rain_km`, `rain_30_n`, `rain_30_e`, `rain_30_s`, `rain_30_w`, `island_wet_frac`, `island_rain_30`$\}$.

- **Station $s$ need not have reported 15 minutes ago.** The earlier value depends only on the other stations' readings at $t - \ell$. A neighbour missing at $t - \ell$ simply drops out of that earlier summary.
- **Reading the sign:** positive means more rain, or more wet stations, than 15 minutes ago. For `dist_rain_km`, negative means the nearest rain has come closer.
- `t2_idw_rain_15`, `t2_idw_trend` and `t2_nbrs_10km` get no change feature: the first two already describe a trend, and the third is a station count.

### `t3_up15_rain`, `t3_up15_wet`, `t3_up30_rain`, `t3_up30_wet`: rain where the storm is coming from

Task 1 (Figure 23) measured, for each season $\sigma$ (coded as `t0_season`), the direction $\phi_\sigma$ that storms move towards (degrees from north, clockwise) and their speed $v_\sigma$:

| Season | Code | $\phi_\sigma$ | $v_\sigma$ |
|---|---|---|---|
| NE monsoon (Dec–Mar) | 0 | 136.4° | 51.4 km/h |
| Inter-monsoon | 1 | 118.2° | 59.1 km/h |
| SW monsoon (Jun–Sep) | 2 | 79.3° | 37.9 km/h |

Following the motion backwards from station $s$ for $\tau$ minutes gives the point that rain reaching $s$ after $\tau$ minutes is over now:

$$
\mathbf{u}_{s}(\tau) = (x_s, y_s) - v_\sigma \frac{\tau}{60}\,(\sin\phi_\sigma, \cos\phi_\sigma), \qquad \tau \in \{15, 30\},
$$

where $\sigma$ is the season of $t$. For example, in the southwest monsoon the 15-minute point is 9.5 km west of the station, and the 30-minute point 19 km west. Stations near that point get Gaussian weights:

$$
k_{sj}(\tau) = \exp\!\Big(-\frac{\lVert (x_j, y_j) - \mathbf{u}_s(\tau)\rVert^2}{2\cdot 4^2}\Big)\;\mathbb{1}\big[\lVert (x_j, y_j) - \mathbf{u}_s(\tau)\rVert \le 10\big]\;\mathbb{1}[j \ne s],
$$

and the features are weighted means over the reporting stations, with the same $\operatorname{mean}_w$ as in Tier 2:

$$
\texttt{t3\_up}\tau\texttt{\_rain} = \operatorname{mean}_{k(\tau)}\big(R^{15}\big), \qquad
\texttt{t3\_up}\tau\texttt{\_wet} = \operatorname{mean}_{k(\tau)}\big(\mathbb{1}[R^{15} > 0]\big).
$$

- **Why $R^{15}$:** the upstream features use the last 15 minutes of rain, the most current picture of where the rain is now.
- **The weights:** $\sigma = 4$ km is roughly the spacing of the stations, and the 10 km cut-off ignores stations whose weight would be below 5%.

### `t3_up30_nbrs`: how many stations are near the 30-minute upstream point

$$
\texttt{t3\_up30\_nbrs}_{s,t} = \sum_j \mathbb{1}[k_{sj}(30) > 0]\; a_{j,t} .
$$

For stations near the western and southern coasts, the 30-minute point often lies over the sea or in Johor, where there are no gauges. There the two `up30` features are 0 because nothing is measured, not because it is dry, and this count tells the model which case it is.

All Tier 3 values are rounded to 4 decimals.

## Model-specific transformations

The features above are what the random forest sees. The logistic regression (`code/models.py`, `build_lr`) first transforms them:

$$
\tilde z = \log(1 + z) \quad \text{for rain amounts and } \texttt{t1\_min\_since\_rain}, \qquad
\tilde z = \operatorname{sign}(z)\,\log(1 + |z|) \quad \text{for the two trend features},
$$

- The log is applied to the features whose names start with `t1_rain`, `t1_max5`, `t1_min_since`, `t2_idw_rain`, `t2_far_rain`, `t2_max_rain`, `t2_rain_30`, `t2_island_rain`, `t3_up15_rain` and `t3_up30_rain`.
- The signed log is applied to the two trend features and to the Tier 3 changes in rain amounts (`t3_d15_idw_rain_30`, `t3_d15_far_rain_30`, `t3_d15_max_rain_30`, `t3_d15_rain_30_*`, `t3_d15_island_rain_30`); changes in shares and in distance stay linear.
- `t0_season` becomes two one-hot columns (the third season is the baseline).
- Every column is divided by its training standard deviation. That standard deviation is estimated on the training table only.

With `nonlinear=True`, two more inputs are added:
- **Bands.** The main rain, wetness, distance and time-since-rain features are cut into bands and one-hot encoded (edges in `BIN_SPLITS`; rain amounts use the label edges).
- **Hour × season.** Each season indicator is multiplied by each of the six hour-of-day sine/cosine terms.

## Row weight

The weight is not a feature, but every model uses it:

$$
w_{s,t} =
\begin{cases}
1/f & \text{if label}_{s,t} = 0 \text{ in the subsampled training table } (f = 0.1),\\
1 & \text{otherwise, and for every validation, test and full-training row.}
\end{cases}
$$

The subsampled table keeps every row with rain in the next 30 minutes and each dry row with probability $f$. Weighting each kept dry row by $1/f$ restores the true rain frequency (inverse-probability weighting).
