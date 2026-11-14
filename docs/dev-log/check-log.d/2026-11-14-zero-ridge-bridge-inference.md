# Zero-ridge bridge inference

The Julia inference bridge now resolves serialized fixed-prior records and verifies expected target coordinates before fitting. Nonempty typed priors and typed empty prior vectors retain their existing path. Empty `Any[]` without expected coordinates and typed priors combined with wire-format expected records are refused. Versioned SD-prior inference remains unsupported.

Focused tests passed 70 assertions on the revised bridge source, including profile and bootstrap inference, zero-ridge table and constructor records, typed priors, typed empty vectors, no-prior fits, missing and mismatched targets, active-prior refusals and versioned SD-prior refusals. The earlier failing runs are retained. The bootstrap is a small replay check, not evidence of interval coverage.

The prior full-package suite and combined current-main run used an earlier bridge source. Full regression on this change remains open. No merge or release follows from the focused result.
