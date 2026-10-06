# Arcs: speed6-20260919 (DRM.jl / DRModels.jl)

| id | arc | ledger | status |
|---|---|---|---|
| S3 | `bench/profile_q4_sections.jl`: section profile of the q=4 phylo ML route at p=100/1000/5000 (inner-Newton factorisations, logdet-P factorisation, kron prior rebuilds, takahashi, `_q4_fd_vcov` cold vs warm), p=2000 baseline 52.9 s, p=100 logLik -256.51; TSV under `bench/results/` | leaf-S3 | DONE 2026-09-19 08:40 (commits fb4161bcc, d53c8bfdc, a734d2b90; complete partition; independently verified) |
| S5a | closed-form logdet P (`sparse_aug_plsm.jl:362`, `fit_q4_sparse_tmb.jl:199-200`, `coevolution_q.jl:268-273`, `locscale_marginal.jl:39-41`, `location_only.jl:194-195`); logdet(Q_cond) once per fit; identity test on random Lambda rtol 1e-12 | leaf-S5 (written) | todo, THIRD (under 2% of the fit wall) |
| S5b | `cholesky!` symbolic reuse with pattern carrier and fresh-cholesky fallback (copy the `gaussian_structured.jl:520-551` / `gaussian_sparse_lss.jl:43-72` pattern) into `sparse_pd_chol` and the ridge path, the GLMM inner modes, `_ls_inner_mode`, `build_M`; fallback count asserted zero | leaf-S5 | todo, FIRST (53% of the p=5000 fit wall, 220 factorisations per fit) |
| S5c | warm `u0` into `_q4_fd_vcov` (`gaussian_bivariate.jl:1348-1360`); Wald vcov rtol 1e-8 vs origin/main | leaf-S5 | todo, SECOND (halves the 228 s post-fit vcov) |
| bench | `bench/head_to_head_q4_scaling.jl` paired with drmTMB, nrep 4, 1 BLAS thread, p=100/1000/5000, before and after | leaf-S5 G5.7 | after S5c |
| verify | Haiku mechanical re-verify; Opus judgment (Noether audit + refute one passed gate); draft PR by the orchestrator; Shinichi's sign-off to land | orchestrator | after bench |

Order re-set 2026-09-19 08:40 by the orchestrator on S3's complete partition (adaptive; recorded for the plan-vs-actual): S5b -> S5c -> S5a. Next measured owners not in the plan: beta-block trace (23%) and Gst/v assembly (17%) at p=5000; handed over, not in this arc.
