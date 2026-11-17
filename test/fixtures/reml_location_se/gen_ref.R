# Generates r_reference.csv: drmTMB 0.8.1 REML fits on deterministic data (no random numbers).
# Run:  Rscript gen_ref.R   (needs drmTMB 0.8.1). The same data are built in
# test_reml_location_se_r_parity.jl; keep the two generators identical.
suppressMessages(library(drmTMB))
mk <- function(name) {
  if (name == "a") {            # 24 rows, one predictor
    i <- 1:24; x <- sin(i); u <- (0.6180339887 * i) %% 1
    data.frame(y = 0.5 + 0.8 * x + exp(1.2 * x) * 1.5 * (2 * u - 1), x = x)
  } else if (name == "b") {     # 40 rows, two predictors
    i <- 1:40; x <- sin(i); z <- cos(1.3 * i); u <- (0.7548776662 * i) %% 1
    data.frame(y = 1 + 0.6 * x - 0.4 * z + exp(0.5 * x - 0.3 * z) * 0.8 * (2 * u - 1), x = x, z = z)
  } else {                      # 100 rows, 20 groups
    i <- 1:100; g <- factor(((i - 1) %% 20) + 1)
    x <- sin(0.7 * i); u <- (0.6180339887 * i) %% 1; ge <- sin(1:20 * 2.3) * 0.8
    data.frame(y = 1 + 0.5 * x + ge[as.integer(g)] + exp(0.5 * x) * 0.9 * (2 * u - 1), x = x, g = g)
  }
}
cases <- list(
  list("a_s1", "a", bf(y ~ x, sigma ~ 1)),
  list("a_sx", "a", bf(y ~ x, sigma ~ x)),
  list("b_sxz", "b", bf(y ~ x + z, sigma ~ x + z)),
  list("c_re_sx", "c", bf(y ~ x + (1 | g), sigma ~ x))
)
rows <- list()
for (cs in cases) {
  f <- drmTMB(cs[[3]], family = gaussian(), data = mk(cs[[2]]), REML = TRUE)
  V <- vcov(f, full = TRUE)
  rows[[length(rows) + 1]] <- data.frame(case = cs[[1]], kind = "logLik", i = "", j = "", value = as.numeric(logLik(f)))
  for (a in rownames(V)) for (b in rownames(V))
    rows[[length(rows) + 1]] <- data.frame(case = cs[[1]], kind = "vcov", i = a, j = b, value = V[a, b])
  cf <- c(coef(f)$mu, coef(f)$sigma)
  for (k in seq_along(cf)) rows[[length(rows) + 1]] <- data.frame(case = cs[[1]], kind = "coef", i = rownames(V)[k], j = "", value = cf[[k]])
}
out <- do.call(rbind, rows)
out$value <- signif(out$value, 15)
write.csv(out, "r_reference.csv", row.names = FALSE)
