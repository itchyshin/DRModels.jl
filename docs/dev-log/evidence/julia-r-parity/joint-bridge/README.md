# Joint missing-predictor R bridge development evidence — issue563

Two Gaussian-response cases (Gaussian/Bernoulli predictor) pass public bridge
adapters and independent mathematical/output checks. **Native parity PASSES**
(2026-10-03, #606): `joint-public-003.json` was regenerated against drmTMB `main`
0eb0467851 (Newton polish on by default); the worst native error is 3.5e-9
(Gaussian imputed SE), against the unchanged 4e-6 bar. The 2026-08-30 receipt it
replaces failed (Gaussian training mean 0.933, Bernoulli `newdata` error, theta
gaps up to 1.0015e-5). It remains in git history; 002 and 001 keep earlier states.
Elapsed003=36.726 seconds including startup and both engines: NOT a warm benchmark.
The checker requires current source/runner hashes; do not relabel old receipts
as evidence for changed sources. Direct reference: joint-direct-bridge-002.toml
(001 retained). Provenance and run notes: `../nongaussian-refresh-20261003/`.

Full native fit RDS artifacts remain in the GPL drmTMB repository, not this MIT
repository. Only generated data/numerical outputs and diagnostics are retained
here. R source remains in drmTMB. Rose's review scope is bounded in rose-review.md.
All programme gates G0–G8 remain open.

Checker invocation from DRM.jl:
```
python3 tools/check_joint_bridge_public_receipt.py docs/dev-log/evidence/julia-r-parity/joint-bridge/joint-public-003.json test/fixtures/joint_missing_predictor/native_reference.toml docs/dev-log/evidence/julia-r-parity/joint-bridge/joint-direct-bridge-002.toml DRMTMB_ROOT DRMODELS_JL_ROOT
```
Add --native to exercise the REQUIRED native gate (passing since 2026-10-03). It is not optional
programme scope. Run test_joint_bridge_public_receipt.py with the same arguments
normally and with python3 -O to exercise the deliberately damaged receipts (24: 21
plus a forged native PASS at +1e-5, a forged PASS just above 4e-6, and a
just-below positive control, added in the PR #934 review). DRMTMB_ROOT and
DRMODELS_JL_ROOT can be any checkouts with the recorded source bytes; the checker
compares repository-relative paths and sha256, not the recorded absolute paths.
