# Focused current-main Julia test runs on Totoro

Source: DRModels.jl current main `973bbb675dc6a10b436acf86175e089e38ec467e`.
The source tar archive SHA-256 is `4b4808c6acef38087e06c8779a563755bc5d2ae8855e3c401165d39bafd1ffb4`.
Each run used an extracted source copy under `~/hsq_work`; no registered worktree was changed.

## Julia 1.10.10

With `JULIA_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`,
`test/test_gaussian_ranef.jl` passed 12/12 and `test/test_tnb2_nan.jl` passed
12/12. Totoro load at launch was about 289. The original raw logs are retained
as gzip files: `gaussian-ranef.log.gz` and `tnb2-nan.log.gz`.

## Julia 1.13.1

With one Julia and BLAS thread, `test/test_temporal_ou.jl`,
`test/test_ls_inv2x2_chol.jl`, and `test/test_tnb2_nan.jl` passed 95, 78, and
12 assertions, respectively, for 185 total. Totoro load at launch was about
299. The Gaussian random-effects file was not included in this Julia 1.13.1
run.

The run warned that the manifest was created with Julia 1.10.10. JET
precompilation emitted an error, but all requested test sets completed and the
Julia command exited zero. Treat this as focused test evidence with a dependency
environment caveat, not as a clean package precompile or full-suite result.
The original raw log is `julia1131-focused-run.log.gz`.

`test/test_temporal_ou.jl` had 95 assertions; `test/test_ls_inv2x2_chol.jl` had
78 across four test sets (12 + 60 + 2 + 4).

## Source and environment identities

The four test files match current-main Git blobs:

| File | SHA-256 |
|---|---|
| `test/test_gaussian_ranef.jl` | `d789688423c0dfa2df5ec1113ef0d21639f987029f5c69545b8b7afd3dd2a3fa` |
| `test/test_tnb2_nan.jl` | `7c48dc928fb6629ad2c51169dc8c286fa85de2bc2c9eeac872ee5b17bc150530` |
| `test/test_temporal_ou.jl` | `460b8bc4dedf2a08684cb6ca04b267d065d37b23201f17e1dadbbe701b2fb6c8` |
| `test/test_ls_inv2x2_chol.jl` | `7143cbe0a0a85e6503ff3ff072642e97102bad3c88adc57e78d73311f0fa5bba` |

The test environment began with a manifest from the Julia 1.10.10 PR #1237
checkout (`pre-resolution-TestManifest-julia110.toml`, SHA-256
`4464382d80dc77ea2935314475ab58c79c4065224b7392fb2b28798dbb493533`).
`Pkg.develop` rewrote the DRModels path to the exact-main source archive. The
manifest present during the Julia 1.13.1 test was
`julia1131-resolved-Manifest.toml`, SHA-256
`24dcd114b002e5e79488fe427ee8fc92bad6601b166b01622a651e8320c387e4`; its
test project SHA-256 was `47c0ee7bdb45889512d12205452577e26e8b3a124c693d68280eaed8280c9520`.
The raw Julia 1.13.1 log reports the version warning and the JET precompile
error. The earlier input checksum is retained separately in
`pre-resolution-manifest.sha256`.

Raw log SHA-256 values:

| Log | SHA-256 |
|---|---|
| Decompressed `gaussian-ranef.log.gz` | `cb8c4b97c5bf677d655c51a77f50d9920dd686a3a53e3f88a893c679df72f9b4` |
| Decompressed `tnb2-nan.log.gz` | `0a9ea2a61874667de9a3ba1f7f2c8169ea30399c460829b161b3e957fd5a4998` |
| Decompressed `julia1131-focused-run.log.gz` | `3d1f33eb0c9fa7bca59a27b0a881fd84f988d6e3060b172acf6930be3f2232ec` |
| Decompressed `julia1131-alias-not-installed.log.gz` | `788900aad7918b08d5a68a4d077eff3a5dd3c02f4341ef5b7cdeef86d4d17162` |

The alias setup log `julia1131-alias-not-installed.log.gz` records a failed
attempt to launch `+1.13.1`; no test ran in that attempt. The successful
Julia 1.13.1 launch used Juliaup alias `+1.13`. Recover a log with
`gzip -dc <file.log.gz>`; the hashes above are of the uncompressed bytes.

These runs do not test the J3a failure reported by DRModels.jl #1247 at
`test/test_j3a_gradient_hessian_vcov.jl:158`. They also do not establish full
package CI, Windows/macOS behavior, the Julia 1.13.3 runner, numerical R–Julia
parity, recovery, inference, or interval calibration.
