# Run from the repository root; load only the preparation helper.
for (expr in parse("tools/bench_bootstrap.R")) {
  if (is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
      is.symbol(expr[[2L]]) && identical(as.character(expr[[2L]]), "prepare_avonet_subset")) eval(expr)
}
stopifnot(exists("prepare_avonet_subset", mode = "function"))
tree <- ape::read.tree(text='(B_b:0,C_c:0.2,(A_a:0.3,D_d:0.4):0);')
dat <- data.frame(Species3=c('A a','D d','C c','B b','unmatched'),
                  Mass=c(1,NA,exp(2),exp(1),5),
                  check.names=FALSE, 'Hand-Wing.Index'=c(1,2,5,3,1),
                  Beak.Length_Culmen=c(9,8,5,7,1),
                  unrelated=c(NA,1,1,1,1))
result <- prepare_avonet_subset(dat,tree,3)
stopifnot(identical(result$tree$tip.label,c('B_b','C_c','A_a')),
          identical(as.character(result$data$species),c('B_b','C_c','A_a')),
          identical(levels(result$data$species),result$tree$tip.label),
          isTRUE(all.equal(result$data$log_mass,c(1,2,0))),
          isTRUE(all.equal(result$data$hand_wing_z,c(0,1,-1))),
          isTRUE(all.equal(result$data$beak_z,c(0,-1,1))),
          all(result$tree$edge.length >= 1e-8))
named <- dat; rownames(named) <- c('first','second','third','fourth','fifth')
stopifnot(identical(rownames(prepare_avonet_subset(named,tree,3)$data), c('fourth','third','first')))
extra <- dat; extra$Mass[2] <- 4
selected <- prepare_avonet_subset(extra,tree,3)
stopifnot(identical(selected$tree$tip.label,c('B_b','C_c','A_a')))
expect_error <- function(expr) stopifnot(inherits(tryCatch({force(expr);NULL},error=identity),'error'))
expect_error(prepare_avonet_subset(dat,tree,4))
bad <- dat; bad$Mass[1] <- 0
stopifnot(identical(tail(prepare_avonet_subset(bad,tree,3)$data$log_mass,1), -Inf))
bad$Mass[1] <- -1
stopifnot(is.nan(suppressWarnings(tail(prepare_avonet_subset(bad,tree,3)$data$log_mass,1))))
bad <- dat; bad[['Hand-Wing.Index']] <- 5
stopifnot(all(is.nan(prepare_avonet_subset(bad,tree,3)$data$hand_wing_z)))
bad <- rbind(dat, dat[1,]); bad$Mass[nrow(bad)] <- 7
stopifnot(tail(prepare_avonet_subset(bad,tree,3)$data$mass,1) == 1)
bad_tree <- tree; bad_tree$edge.length[1] <- NA_real_
stopifnot(anyNA(prepare_avonet_subset(dat,bad_tree,3)$tree$edge.length))
root_tree <- tree; root_tree$root.edge <- 0
root_result <- prepare_avonet_subset(dat,root_tree,3)
stopifnot(identical(root_result$tree$root.edge, 0),
          identical(names(root_result$data), c('species','mass','hand_wing','beak','log_mass','hand_wing_z','beak_z')),
          identical(rownames(root_result$data), c('4','3','1')))
expect_error(prepare_avonet_subset(dat,tree,-1))
zero_warning <- NULL
zero <- withCallingHandlers(prepare_avonet_subset(dat,tree,0), warning=function(w) {
  zero_warning <<- conditionMessage(w); invokeRestart('muffleWarning')
})
stopifnot(grepl('drop all tips',zero_warning), identical(zero$tree, list(edge.length=numeric())), nrow(zero$data)==0)
cat('zero count outcome: ape warning, empty edge-length list, zero data rows\n')
two <- prepare_avonet_subset(dat,tree,2)
stopifnot(identical(rownames(two$data), c('4','3')))
one <- prepare_avonet_subset(dat,tree,1)
stopifnot(identical(as.character(one$data$species),'B_b'), is.nan(one$data$hand_wing_z), identical(rownames(one$data), '4'))
cat('PASS: tree order, four-field complete cases, transforms, clamping, and edge outcomes\n')
