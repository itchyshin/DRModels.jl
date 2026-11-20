- **Bridge accepts drmTMB's pinned Binomial animal `recov` label (#1401).**
  An unlabelled `animal(1 + x | group)` fit pins the intercept–slope correlation
  at 0, so its blocks are `mu` and two `sdtarget_*` SD regressions. drmTMB still
  sends `coef_labels["recov"] = "<group>:atanh_rho"`. The bridge now drops that
  one name and echoes the blocks that exist. A different `recov` spelling, and
  every other unknown dpar, still abort. The labelled form keeps its real
  `:recov` coefficient.
