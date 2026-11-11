# Paper draft

Workshop draft (ACM sigplan, 10pt, two columns; EuroMLSys-style 6-page limit).

```bash
python paper/figures/make_figures.py --results results --out paper/figures   # figures from result CSVs
cd paper && latexmk -pdf main.tex
```

`acmart.cls` and `ACM-Reference-Format.bst` are bundled (built from the acmart
GitHub sources) so the draft builds without a full TeX Live; Overleaf has both.
Without the Libertine fonts the local build falls back to Latin Modern, so the
page count on Overleaf can differ slightly.

Figure inputs: `results/paper/breakdown_*.csv` (`analysis/episode_breakdown.py`
over the G5 `given` and G6 `nobatch` cells), `results/b_feasibility`,
`results/b2a/*/interference_mps`, `results/b_batch`, `results/b3`,
`results/g6/*/analysis`. Bibliography entries marked TODO need checking.
