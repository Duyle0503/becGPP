# Sample output

A representative subset of the output of `becgpp --mode scan_all --scan_profile full`
(one full production run on a single GPU, code version `becGPP_1.0`). It is here so
that a new user can see the exact I/O format without running the code first.

```
validation.csv                     analytic gate: relative errors (Coulomb, LLL, flat-top TF)
scan/scan_manifest.json            run metadata: code version, device, profile, wall time
scan/scan_singles.csv              the 8 showcase ground states, full diagnostic columns
scan/scan_sweep_rotating_Omega.csv Omega sweep: Lz, Nv, w_LLL, R90, TF radius
scan/scan_convergence.csv          grid convergence: E, R90, virial, points-per-R90 vs dx
scan/scan_benchmark.csv            GPU throughput: ms/iter and ns/point/iter vs N
fig/scan_2D_rotating_lattice_dens.png     density of a 19-vortex Abrikosov lattice
fig/scan_2D_cubicquintic_flattop_TF.png   GPE density vs flat-top TF (rho0 = 0.75)
fig/scan_3D_rotating_oblate_dens_slices.png  oblate rotating self-gravitating cloud
```

CSV columns follow the diagnostic set documented in the top-level README. The full
run also produces per-case density/phase/TF figures for every showcase and sweep
point (omitted here to keep the repository small).
