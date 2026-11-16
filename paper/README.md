# Paper draft

IEEE conference format, 10pt, two columns, 9 pages excluding references,
double-blind.

```bash
python analysis/anatomy.py \
  --set "replay_N1=results/g2_flash/20261024-204239/N1_r0" \
  --set "replay_N4=results/g2_flash/20261024-193111/N4_r0" \
  --set "replay_N8_ts=results/g5/*/given/N8_r*" \
  --set "replay_N8_mps=results/g6/*/nobatch/N8_r*" \
  --set "replay_N16_mps=results/g6/*/nobatch/N16_r*" \
  --out results/paper/anatomy
python analysis/bounds.py
python paper/figures/make_figures.py
cd paper && latexmk -pdf main.tex
```

`main.tex` uses `IEEEtran` (Overleaf, or a full TeX Live). Without the class it
falls back to a two-column `article` so the text and figures can be checked;
page counts are only valid with the real class.

`\todo{...}` marks text that waits for an experiment (ids as in the top-level
README); `\num{...}` marks numbers copied from a script's output, to re-check
whenever the scripts are rerun. Keep author names, repository URLs and machine
names out of this directory.
