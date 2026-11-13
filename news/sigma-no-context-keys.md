- **`sigma(fit)` no longer returns internal fit context.** For a Binomial fit it
  returned `Dict(:trials => ...)`, and for a cumulative-logit fit the linear
  predictor and cut points, as if they were scale parameters. Both now return an
  empty `Dict`, as the docstring says for families without a fitted scale. A
  Beta-binomial fit returns its `sigma` vector instead of a `Dict` that also
  held the trials. The R bridge `sigma` entry follows.
