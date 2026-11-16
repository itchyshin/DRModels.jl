# Codecov CI — Julia coverage collection + documented upload

Personas: **Grace** (CI), **Rose** (claim-vs-evidence). No spawned subagents.

## Change

Narrow Codecov wiring on the existing authorised test lane:

- `.github/workflows/CI.yml` — `julia-runtest` coverage only on Julia `1`;
  `julia-processcoverage@v1` writes `lcov.info` per shard; artefacts named
  `coverage-lcov-julia1-shard{1..4}`; new non-required `codecov` job uploads
  all four files in one `codecov/codecov-action@v5` call with `use_oidc: true`.
- `codecov.yml` — informational project/patch status; ignore test/docs/bench/tools/report;
  carryforward flag `julia-linux-1`.
- No `src/` / test / fixture / README badge changes.

## What this is not

| State | This PR |
|---|---|
| Configured | yes |
| Locally coverage-tested | no (`julia` missing in this image) |
| CI running | pending after push |
| Upload accepted | NOT_VERIFIED until the `codecov` job log says so |
| Report visible on Codecov | NOT_VERIFIED; dashboard URL is not proof |
| Default-branch (`main`) functional | no — this PR does not merge; CI has no `push` to `main`; first `main` upload is the next `schedule` / `workflow_dispatch` **after** owner merge |

## Auth

- GitHub Actions OIDC (`permissions.id-token: write` on the `codecov` job only).
- No `CODECOV_TOKEN`, no `GITHUB_TOKEN` passed as a Codecov token.
- Same-repo `pull_request` only (fork PRs skip the upload job).
- `fail_ci_if_error: true` on `codecov` so an auth miss is visible; that job is **not** part of `ci-ok`.

## Owner prerequisites if OIDC is rejected

1. Sign in at https://app.codecov.io/github/itchyshin as the GitHub owner of `itchyshin/DRModels.jl`.
2. Activate `itchyshin/DRModels.jl` (Codecov GitHub App: https://github.com/apps/codecov, access to this repo). App is **not** required for the upload itself; it is required for reliable PR comments under GitHub API rate limits.
3. Public-repo token policy lives on Codecov's **Global Upload Token** tab (admin). OIDC is the intended auth; if Codecov rejects the OIDC token, the owner may add repository secret `CODECOV_TOKEN` from that repo's Codecov Configuration. Do not mint, paste, or substitute `GITHUB_TOKEN`.
4. Do **not** add the `codecov` check to branch protection until an upload has been accepted.

## Double-count guard

One OS (ubuntu), one Julia version (`1`), one combined upload of four complementary shards. Julia 1.10 nightly and release-platform macOS/Windows do not upload.
