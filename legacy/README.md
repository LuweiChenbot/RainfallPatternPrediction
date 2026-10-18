# Legacy: the first Task 1 pipeline

Superseded by `task1/` and kept for reference only; nothing in the current project depends on these files.

- `task1_analysis.py`: Spark tables of station-month and station-year rainfall, coverage and extremes, written to
  `outputs/task1/`
- `project_common.py`: Spark settings and helpers for `task1_analysis.py`
- `plot_task1.py`: the first eight Task 1 figures, drawn from those tables
- `task1_analysis.md`: the design notes and run commands of this pipeline
- `requirements-task1.txt`: its packages

The paths in these files are those of the old layout (`code/...`); run them from the repository root as
`python legacy/task1_analysis.py ...`.
