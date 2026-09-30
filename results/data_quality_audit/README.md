# Data-quality audit tables

These small CSV files were produced by the latest verified run of
`code/rainfall_audit.py` on 2026-09-30. They contain summaries only; no raw
rainfall dataset is included.

- `yearly_quality.csv`: annual completeness, validity, and update-time checks
- `duplicate_summary.csv`: original station-time duplicate count
- `duplicate_examples.csv`: examples when duplicates exist (header only here)
- `units_and_reading_types.csv`: exact annual units and reading types
- `station_metadata.csv`: metadata versions and canonical display names
- `station_coverage.csv`: station-year coverage diagnostics
- `coverage_by_year.csv`: annual coverage summary
- `early_2017_timestamps.csv`: sample of the legacy 2017 timestamp rhythm
