DSA5208 Rainfall Report — LaTeX source package
Revision: 18 October 2026

Files
  DSA5208_Rainfall_Report.tex         Main LaTeX source
  DSA5208_Rainfall_Report_source.txt  Identical source in plain-text form
  references.bib                    Bibliography database
  figures/                          All twelve figure assets

Compile from this folder with XeLaTeX and BibTeX:
  xelatex DSA5208_Rainfall_Report.tex
  bibtex DSA5208_Rainfall_Report
  xelatex DSA5208_Rainfall_Report.tex
  xelatex DSA5208_Rainfall_Report.tex

In Overleaf, upload the entire package, choose the .tex file above as the main
document, and select XeLaTeX. Bibliography processing is normally automatic.
Keep the figures folder and references.bib alongside the main source.

This package intentionally contains no compiled PDF. Static source checks are
not a substitute for checking the final page layout in your LaTeX environment.
The app's single-file compiler could read the revised preamble but stopped at
the first external figure; that preview does not support project attachments.
Compile the complete package to resolve figures and the BibTeX bibliography.

Revision scope
  Abstract and its contents entry removed.
  Verified data, software, meteorological and methodological references added.
  Figure captions use automatic numbering and a separate 9-point sans-serif
  style. The duplicated Figure 8 reference for daily persistence is corrected.
  Figure placement, sizes and spacing are adjusted to reduce large blank areas.
  Mathematical expressions use standard LaTeX inline/display notation.
  Other report wording and numerical results are preserved.

Existing content placeholders, including Windows TBD, the empty data-quality
table heading and the empty appendices, have not been completed or deleted.
