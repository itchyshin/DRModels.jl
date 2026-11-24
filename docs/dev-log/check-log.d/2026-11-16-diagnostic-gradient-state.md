# Diagnostic-gradient cache check

The correlated-slope Gamma fixture reproduced a report mismatch on candidate `cf7477ab`: repeated gradients at identical parameters changed as the retained AGHQ warm-start cache changed. A new regression test on that source produced four failures in eight assertions. The stored source and log are in the release evidence folder.

On a separate source assembled from base `cc178a7d`, the four-file diagnostic patch and its focused test, 44 new assertions and 13 existing report assertions passed on Totoro with Julia 1.12.6 and one thread. Source-file hashes matched after the run. The copied environment was not independently hashed before and after. A separate serial test forced a nonfinite mode through the actual AGHQ routine and passed six assertions for diagnostic versus ordinary reset counting.

The complete final focused test file then passed all 50 assertions on the same patched source and environment, in 103.46 seconds. Its SHA-256 was checked after the run; the prior environment-hash limitation still applies.

On three fixtures, separate original and candidate runs printed identical parameter vectors, log likelihoods and convergence flags. Their command and later source/environment records are retained; the fit logs themselves do not print source paths. This comparison does not establish equality for other data or certify gradient accuracy.

The full suite and hosted CI on the final source remain pending. The wrapper changes the truthful internal type label returned by `bridge_diagnostics.route`; it does not change the public model or integration choice. Diagnostics are repeatable from the same saved cache, but other objective calls can still warm the live cache. The existing global reset counter can still be affected by concurrent ordinary evaluations.
