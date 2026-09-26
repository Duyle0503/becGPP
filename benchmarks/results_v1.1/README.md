# becGPP v1.1.0 — benchmark results used in the paper

Raw outputs of `benchmarks/kaggle_v11_step{2,3,4}_*.py`, run on a Kaggle
Tesla T4 (see `hardware.json` in each folder), 2026-09-25. Every number in
Secs. 5–7 of the paper is read from these files; `make_paper_numbers.py`
prints them in LaTeX form.

Note: in `step3/ref_mass_radius.csv` the runs with beta2 = 32, 128, 512 were
box-limited (virial residual > 1e-3; the box was sized from the Thomas–Fermi
radius) and are not used in the paper. The script now sizes the box from an
interpolated R99 estimate.
