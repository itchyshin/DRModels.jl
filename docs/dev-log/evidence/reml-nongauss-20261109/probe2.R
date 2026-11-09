.libPaths(c("~/drm_campaign_20261108/Rlib", .libPaths()))
suppressMessages({library(drmTMB); library(glmmTMB)})
sdof <- function(f) as.numeric(f$sdpars[[1]][1]) 
gen <- function(cell, seed) {
  set.seed(seed); G<-cell$G; m<-cell$m; g<-factor(rep(1:G,each=m)); x<-rnorm(G*m)
  if (cell$slope) u <- rnorm(G,0,cell$sd)[g]*x else u <- rnorm(G,0,cell$sd)[g]
  if (cell$fam=="gaussian") y <- -0.3+0.8*x+u+rnorm(G*m) else y <- rbinom(G*m,1,plogis(-0.3+0.8*x+u))
  data.frame(y,x,g)
}
cells <- list(
 A=list(fam="binomial",G=15,m=8, sd=1.0,slope=FALSE),
 B=list(fam="binomial",G=40,m=6, sd=0.6,slope=FALSE),
 C=list(fam="binomial",G=20,m=10,sd=0.8,slope=TRUE),
 D=list(fam="gaussian",G=10,m=6, sd=0.8,slope=FALSE),
 E=list(fam="binomial",G=8, m=20,sd=1.0,slope=FALSE))
out <- list()
for (nm in names(cells)) { cl <- cells[[nm]]; for (s in 1:40) {
  d <- gen(cl, 1000*match(nm,names(cells))+s)
  fm <- if (cl$slope) bf(y~x+(0+x|g)) else bf(y~x+(1|g))
  fam <- if (cl$fam=="gaussian") gaussian() else binomial()
  r <- lapply(c(FALSE,TRUE), function(rm) tryCatch(drmTMB(fm,data=d,family=fam,REML=rm),error=function(e) NULL))
  gl <- tryCatch({ff <- if (cl$slope) y~x+(0+x|g) else y~x+(1|g)
     glmmTMB(ff,data=d,family=if(cl$fam=="gaussian") gaussian() else binomial(),REML=TRUE)},error=function(e) NULL)
  gsd <- if(!is.null(gl)) as.numeric(attr(VarCorr(gl)$cond[[1]],"stddev")[1]) else NA
  one <- function(f) if(is.null(f)) c(NA,NA) else c(sdof(f), as.numeric(isTRUE(is_converged(f))))
  o <- c(one(r[[1]]), one(r[[2]]), gsd, if(!is.null(r[[2]])) as.numeric(r[[2]]$estimator_exact) else NA)
  out[[length(out)+1]] <- data.frame(cell=nm, seed=s, sd_ml=o[1], conv_ml=o[2], sd_reml=o[3], conv_reml=o[4], sd_glmmTMB_reml=o[5], exact=o[6])
}}
R <- do.call(rbind,out)
write.csv(R,"/tmp/remlng_out.csv",row.names=FALSE)
for (nm in names(cells)) { z<-R[R$cell==nm,]; ok<-complete.cases(z)&z$sd_ml>1e-3
 cat(sprintf("%s fam=%s n_ok=%d med_sd_ml=%.3f med_sd_reml=%.3f frac_reml>ml=%.2f conv_reml=%.2f conv_ml=%.2f max|reml-glmmTMB|=%.2e exact=%s\n", nm, cells[[nm]]$fam, sum(ok), median(z$sd_ml[ok]), median(z$sd_reml[ok]), mean(z$sd_reml[ok]>=z$sd_ml[ok]), mean(z$conv_reml,na.rm=TRUE), mean(z$conv_ml,na.rm=TRUE), max(abs(z$sd_reml-z$sd_glmmTMB_reml),na.rm=TRUE), paste(unique(z$exact),collapse=","))) }
cat("true sd: A1.0 B0.6 C0.8 D0.8 E1.0\n")
