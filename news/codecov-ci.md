- **CI collects Julia test coverage and uploads it to Codecov.** The existing
  four Linux shards still run the authorised suite (`DRM_TEST_SHARD`); only the
  current-stable Julia 1 legs write `lcov.info`. A separate non-required
  `codecov` job merges those four files and uploads them with GitHub Actions
  OIDC. No README badge until an accepted upload is verified.
