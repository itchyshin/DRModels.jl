.libPaths(c("~/drm_campaign_20261108/Rlib", .libPaths()))
suppressMessages(library(drmTMB)); cat("drmTMB", as.character(packageVersion("drmTMB")), "\n")
d <- read.csv("pn_data.csv", stringsAsFactors = FALSE)
nd <- data.frame(x = c(0.3, -0.5, 1.0, 0.0), f = c("c", "a", "d", "b"))
res <- list()
for (fam in c("poisson", "gamma", "beta")) {
  y <- d[[paste0("y_", fam)]]; dd <- cbind(d, y = y)
  fit <- switch(fam,
    poisson = drmTMB(bf(mu = y ~ x + f), family = poisson(), data = dd),
    gamma = drmTMB(bf(mu = y ~ x + f, sigma = ~ 1), family = Gamma(link = "log"), data = dd),
    beta = drmTMB(bf(mu = y ~ x + f, sigma = ~ 1), family = beta_family(), data = dd))
  for (lab in list(single = 1, subset = 2:3, reordered = 4:1)) {
    l <- predict(fit, newdata = nd[lab, ], dpar = "mu", type = "link")
    r <- predict(fit, newdata = nd[lab, ], dpar = "mu", type = "response")
    for (k in seq_along(lab)) cat(sprintf("%s,%s,%d,%.12g,%.12g,%.12g\n", fam, c("single","subset","reordered")[length(lab)==c(1,2,4)], lab[k], l[k], r[k], as.numeric(logLik(fit))))
  }
}
