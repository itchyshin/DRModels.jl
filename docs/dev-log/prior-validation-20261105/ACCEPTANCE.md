# Gates: finish Julia prior validation

OWNS: tools/prior-validation-20261105/ docs/dev-log/prior-validation-20261105/ docs/dev-log/after-task/2026-11-05-prior-timeout-validation.md docs/dev-log/check-log.d/2026-11-05-prior-timeout-validation.md

- [x] G0: owner approved diagnosis and validation in the isolated prior lane
  EVIDENCE: 2026-11-05 user continuation; lease granted; no merge or release.
- [x] G1: timeout identified using retained stack and measured baseline timing
  EVIDENCE: SIGTERM at test_crossed_tweedie.jl:28; baseline CI testset7351.1seconds, job teststep10872seconds; timing receipts retained.
- [x] G2: candidate and disposable timing runner match the declared source and complete shard partition
  EVIDENCE: 949 source/test/project files match remote candidate; only disposable runner instrumentation differs; native dry-run discovers554files, partition70/70/69/69/69/69/69/69 is disjoint and complete.
- [x] G3: all eight default-suite shards finish, with every selected file accounted for and no unclassified failure
  CHECK: python3 tools/prior-validation-20261105/verify.py docs/dev-log/prior-validation-20261105/results docs/dev-log/prior-validation-20261105/expected-partition.json
  EXPECT: FULL_DEFAULT_SUITE_PASS
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/.codex/worktrees/drmodels-fixed-priors-20261105/DRM.jl; path=8d424342f38b/44 entries; output=NEGATIVE_CONTROLS_PASS 6/6; scientific-notation timestamps accepted | FULL_DEFAULT_SUITE_PASS 8/8 shards; 554/554 selected files; JET executed; opt-in parity off
- [x] G4: any regression or environment failure independently diagnosed and corrective checks complete
  EVIDENCE: no native test failure; all source and assertions unchanged; earlier90minute timeout explained by113m40crossedTweedie file;6verifiernegativecontrols pass.
- [x] G5: independent review, reports and local closeout commit complete
  EVIDENCE: independent scopedREADY with no remaining findings; final reports/checklogs/provenance present in bothrepos; cleanup andPR1651 protected; local commits only, no push merge release.
