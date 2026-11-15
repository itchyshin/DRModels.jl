- **`lrtest(...; mixture = true)` matches a random-effect SD by its covariance structure and grouping.**
  A one-group `sd(g) ~ 1` is stored as `sd:(Intercept)`, two groups as
  `sd:g: (Intercept)`, and a plain `(1 | g)` as `resd:g`. Those are the same
  iid component, so a nest that only adds one group is accepted. A phylogenetic
  intercept (`phylo(1 | species)` or `sd_phylo`) is a different component from
  an iid intercept on that species, and the same split applies to a spatial,
  animal, or relmat intercept. A swapped grouping, such as `(1 | g)` against
  `(1 | h) + (1 | k)`, is still refused.
