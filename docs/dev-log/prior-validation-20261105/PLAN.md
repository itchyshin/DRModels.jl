# Finish fixed-prior Julia validation

Reader: the maintainer reviewing the isolated prior branch. Source baseline is
24ac6ca; prior implementation is 07b25dc4; evidence checkpoint is 60175d6e.

The earlier unsharded run stopped at 90 minutes inside test_crossed_tweedie.jl.
The exact baseline's completed CI job 127349180304 measured that testset at
7,351.1 seconds, within a 10,872-second shard test step. The old total budget
was shorter than this one unchanged test on the measured CI host. Host timings
are not interchangeable; the new run will measure each selected file directly.

Run the default package suite on Totoro with its existing eight-way positional
sharding, Julia 1.10.10, four Julia threads per shard and one BLAS thread. Estimate
150 minutes wall time; stop each shard at 170 minutes. At most 32 Julia threads
are available to these eight test workers, below the 150-core cap. Available
memory was 228 GB before launch. No run exceeds three hours.

Use the frozen candidate project and dependencies in a disposable validation
copy. Only its runner receives stderr timing around the existing include call.
Keep all assertions, tolerances, source, include order and existing exclusions.
Capture the exact discovery partition and verify each selected file finishes
exactly once across the eight shards. Keep the separately gated JET and parity
status explicit. No CI workflow edit, scientific campaign, merge or release.

If a shard fails, preserve its result, reproduce the failing file at the exact
baseline on the same host, and fix only a demonstrated prior regression. Do
not turn a timeout, missing completion marker or failed assertion into a pass.
After execution, obtain independent review, record the check log and after-task
report, and make scoped local commits. Existing config changes stay unstaged.
