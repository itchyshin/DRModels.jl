# Two-intercept oracle provenance

Shinichi confirmed MIT terms for the identified oracle scripts and fixtures. The adjacent fixture notice records the exact Julia and upstream numerical artifact hashes and retains the unknown original converter and SciPy version. Its MIT notice matches the package license.

Both upstream oracle manifests and verifier scripts passed. The replacement converter reproduced `prior_table_oracle.jl` byte for byte with SHA-256 `b64bf2cfaa027996e40865ff1d20cf5be4042d617349ad56ea8bf95d0ffc77b6`.

The complete Julia suite passed on unmerged combined tree `45de93dd9ca823b8516a7e869e1e7f1f3273e9dd`, which includes parent `92c0a2b4` and main `8d0bd9ea5`; its logs retain warnings and skips, including R parity. That run preceded the present provenance-only additions. Checks on the new commit remain open.
