# Report template

Edit `report.tex`; the five benchmark figures are already included and copied
into `figures/`. Replace the author and course placeholders, fill in the TODOs,
and report the hardware, software versions, correctness checks, and precision
settings from your actual run.

Compile from this folder with a LaTeX distribution installed:

```powershell
pdflatex report.tex
pdflatex report.tex
```

The second pass resolves figure references. To refresh the copied figures after
regenerating the plots, copy the updated PDFs from `../Visuals/images/` into
`figures/`.
