"""Latent-based per-Z-slice registration (see registration/README.md).

Core library: affine (2x3 similarity math + graph solves), graph_tv (fused-
lasso TV solver), features, perturb (corruption generators), register,
enhance, evaluate/evaluate_enhanced. Real-data tools: find_discont (codec gap
mining), measure_drift (whole-stack drift). Round-2 experiment drivers live
in experiments/ (formerly the regis2/ package; results in README.md and
THX10_GAPS_DRIFT.md).
"""
