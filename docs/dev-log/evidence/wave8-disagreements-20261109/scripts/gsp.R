cc <- read.csv("data/coords.csv"); co <- as.matrix(cc[, c("easting","northing")]); rownames(co) <- cc$id
D <- as.matrix(dist(co)); pos <- D[upper.tri(D)]; rg <- median(pos[pos > 0]); K <- exp(-D / rg); diag(K) <- diag(K) + 1e-6
cat("median distance (R fixed range):", rg, "\n")
write.table(K, "data/K_fixedrange.csv", sep = ",", row.names = FALSE, col.names = FALSE)
d <- read.csv("data/data_gaussian_spatial.csv"); cat("data g order == ids order:", identical(unique(d$g), cc$id), "\n")
