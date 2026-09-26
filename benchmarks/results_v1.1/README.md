# becGPP v1.1.0 — benchmark results used in the paper

Raw outputs of `benchmarks/kaggle_v11_step{2,3,4}_*.py`, run on a Kaggle
Tesla T4 (see `hardware.json` in each folder), 2026-09-25. Every number in
Secs. 5–7 of the paper is read from these files; `make_paper_numbers.py`
prints them in LaTeX form.

`step3/` is the second run of the step-3 script (2026-09-26), after the box
sizing was changed from the Thomas–Fermi radius to an interpolated R99 estimate;
in the first run the beta2 = 32, 128, 512 points were box-limited.

Note on `lll_param`: the CSV column was written by 1.1.0 with the incorrect gap
`2(1-Omega)`; the paper quotes `beta2*peak/(1+Omega)` (1.1.1 definition),
recomputed from the `peak` column.
