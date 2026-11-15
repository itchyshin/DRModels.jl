- **`lrtest(...; mixture = true)` matches a random-effect SD by its grouping.**
  A one-group `sd(g) ~ 1` is stored as `sd:(Intercept)`, two groups as
  `sd:g: (Intercept)`, and a plain `(1 | g)` as `resd:g`. Those are the same
  component, so a nest that only adds one group is accepted. The same rule
  applies to `sd_phylo` and a phylogenetic intercept. A swapped grouping,
  such as `(1 | g)` against `(1 | h) + (1 | k)`, is still refused.
