# Generates the lane-N4 fixtures and the drmTMB oracle values recorded in
# test_lane_n4_offset_slopes.jl. Own simulated data; drmTMB is only RUN, never vendored.
# Rscript gen_oracle.R   (drmTMB 0.7.1 @ 0eb0467)
suppressMessages(library(drmTMB))
set.seed(20261011)
## 1. Poisson rate model: offset(log exposure) + (1 | g)
G <- 30; m <- 8; g <- rep(seq_len(G), each = m); n <- G * m
x <- rnorm(n); expo <- round(runif(n, 0.5, 6), 2); u <- rnorm(G, 0, 0.6)
y <- rpois(n, expo * exp(0.4 + 0.3 * x + u[g]))
rate <- data.frame(y = y, x = x, logexp = log(expo), g = paste0("g", g))
write.csv(rate, "rate.csv", row.names = FALSE, quote = FALSE)
## 2. growth data: 14 groups x 7 times
G <- 14; m <- 7; g <- rep(seq_len(G), each = m); n <- G * m
time <- rep(seq(0, 1.2, length.out = m), G); diet <- rep(c("d1", "d2"), length.out = G)[g]
u0 <- rnorm(G, 0, 3); u1 <- rnorm(G, 0, 4); d <- rnorm(G, 0, 0.4)
mu <- 40 + 60 * time + 5 * (diet == "d2") + u0[g] + u1[g] * time
sg <- exp(1.0 + 0.3 * time + d[g])
grow <- data.frame(g = paste0("g", g), time = time, diet = diet)
grow$wgauss <- mu + rnorm(n, 0, sg)
lmu <- log(40) + 0.9 * time + 0.15 * (diet == "d2") + rnorm(G, 0, 0.15)[g] + rnorm(G, 0, 0.2)[g] * time
grow$wgamma <- rgamma(n, shape = 1 / 0.2^2, scale = exp(lmu) * 0.2^2)
grow$wstud <- mu + 2.5 * rt(n, df = 5)
write.csv(grow, "growth.csv", row.names = FALSE, quote = FALSE)
grow$g <- factor(grow$g); grow$diet <- factor(grow$diet); rate$g <- factor(rate$g)
rep1 <- function(tag, f) { cat(tag, "logLik", format(logLik(f), digits = 10), "conv", f$opt$convergence, "\n"); print(unlist(coef(f))) }
f1 <- drmTMB(drm_formula(y ~ x + offset(logexp) + (1 | g)), family = poisson(), data = rate); rep1("POIS_OFFSET_RE", f1); print(summary(f1)$sdpars)
f2 <- drmTMB(drm_formula(wgauss ~ time + diet + (1 + time | g), sigma ~ time + (1 | g)), family = gaussian(), data = grow); rep1("GAUSS_RS_SIGRI", f2); print(summary(f2))
f3 <- drmTMB(drm_formula(wgamma ~ time + diet + (1 | g) + (0 + time | g), sigma ~ 1), family = Gamma(link = "log"), data = grow); rep1("GAMMA_INDEP", f3); print(summary(f3))
f4 <- drmTMB(drm_formula(wstud ~ time + diet + (1 | g) + (0 + time | g), sigma ~ 1), family = student(), data = grow); rep1("STUDENT_INDEP", f4); print(summary(f4))
