# Independent, standalone dense Gaussian phylogenetic SD-regression fit oracle.
# Authored after preliminary package comparisons, not a pre-comparison freeze.
# No package implementation code used.
set.seed(17822L)
n_tip <- 16L; n_rep <- 5L; n <- n_tip*n_rep
# 16-tip balanced covariance: pairs at .5, sister pairs at .25, other quartets at 0.
K <- diag(n_tip)
for (i in seq_len(n_tip)) for (j in seq_len(n_tip)) if(i!=j) {
  if ((i-1L)%/%4L == (j-1L)%/%4L) K[i,j] <- if ((i-1L)%/%2L == (j-1L)%/%2L) .5 else .25
}
group <- rep(seq_len(n_tip),each=n_rep)
z <- seq(-1.5,1.5,length.out=n_tip)
x <- rep(c(-1.2,-.55,.1,.7,1.3),times=n_tip) + .09*rep(sin(seq_len(n_tip)),each=n_rep)
X <- cbind(1,x); S <- X; H <- cbind(1,z)
true <- c(.8,.45,-.4,.22,-.15,.35)
tip_draws <- drop(t(chol(K)) %*% rnorm(n_tip)); residual_draws <- rnorm(n)
y <- drop(X%*%true[1:2]) + exp(drop(H%*%true[5:6]))[group]*tip_draws[group] + exp(drop(S%*%true[3:4]))*residual_draws
prior_index<-c(2L,4L);prior_mean<-c(.1,-.2);prior_sd<-c(.7,.5)
calc <- function(t,gradient=TRUE) {
  E <- exp(2*drop(S%*%t[3:4])); a<-exp(drop(H%*%t[5:6])); C<-K[group,group]*outer(a[group],a[group]); V<-C+diag(E)
  ch<-chol(V); r<-y-drop(X%*%t[1:2]); u<-backsolve(ch,forwardsolve(t(ch),r)); nll<-.5*(n*log(2*pi)+2*sum(log(diag(ch)))+sum(r*u)); P <- -sum(dnorm(t[prior_index],prior_mean,prior_sd,log=TRUE)); value<-nll+P
  if (!gradient) return(value)
  A<-chol2inv(ch); B<-A-tcrossprod(u); g<-numeric(6);g[1:2]<- -drop(crossprod(X,u))
  for (i in 1:2) g[2+i]<-sum(diag(B)*E*S[,i])
  for (i in 1:2) {q<-H[group,i];g[4+i]<-.5*sum(B*C*outer(q,q,'+'))}
  g[prior_index]<-g[prior_index]+(t[prior_index]-prior_mean)/prior_sd^2
  list(value=value,gradient=g,loglik=-nll,penalty=P)
}
starts<-rbind(c(.7,.35,-.6,.1,-.3,.2),c(.3,.9,-.1,-.1,.2,-.1),c(1.2,.1,-.8,.4,-.8,.6),c(.8,.45,-.4,.22,-.15,.35))
fits<-lapply(seq_len(nrow(starts)),function(i) optim(starts[i,],function(t) calc(t)$value,function(t) calc(t)$gradient,method='L-BFGS-B',lower=rep(-5,6),upper=rep(5,6),control=list(maxit=2000,factr=100,pgtol=1e-9)))
fits<-lapply(fits,function(f) optim(f$par,function(t) calc(t)$value,function(t) calc(t)$gradient,method='BFGS',control=list(maxit=3000,reltol=1e-15)))
fits<-lapply(fits,function(f) { for (iteration in 1:3) { h<-1e-5; HH<-sapply(seq_along(f$par),function(k){d<-numeric(length(f$par));d[k]<-h;(calc(f$par+d)$gradient-calc(f$par-d)$gradient)/(2*h)}); f$par<-f$par-drop(solve(HH,calc(f$par)$gradient)) }; f$value<-calc(f$par)$value; f })
f<-fits[[which.min(vapply(fits,`[[`,numeric(1),'value'))]]
theta<-f$par;deriv_step<-1e-5
Hess<-sapply(seq_along(theta),function(i){d<-numeric(length(theta));d[i]<-deriv_step;(calc(theta+d)$gradient-calc(theta-d)$gradient)/(2*deriv_step)})
fdg<-sapply(seq_along(theta),function(i){d<-numeric(length(theta));d[i]<-deriv_step;(calc(theta+d,FALSE)-calc(theta-d,FALSE))/(2*deriv_step)})
eig<-eigen((Hess+t(Hess))/2,symmetric=TRUE,only.values=TRUE)$values
out<-list(schema='public-tree-phylo-fit-oracle-v1',independence_caveat='Created after preliminary package comparisons.',seed=17822L,n_tip=n_tip,n_rep=n_rep,K=K,group=group,x=x,z=z,y=y,tip_draws=tip_draws,residual_draws=residual_draws,generating_theta=true,prior_index=prior_index,prior_mean=prior_mean,prior_sd=prior_sd,starts=starts,fit_values=vapply(fits,`[[`,numeric(1),'value'),fit_codes=vapply(fits,`[[`,integer(1),'convergence'),fit_thetas=lapply(fits,`[[`,'par'),theta=theta,objective=f$value,data_loglik=calc(theta)$loglik,prior_penalty=calc(theta)$penalty,gradient=calc(theta)$gradient,hessian=Hess,hessian_eigenvalues=eig,hessian_condition=max(eig)/min(eig),central_step=deriv_step,gradient_fd_abs=max(abs(fdg-calc(theta)$gradient)))
cat('DIAGNOSTIC theta',theta,'score',max(abs(out$gradient)),'eig',eig,'values',out$fit_values,'codes',out$fit_codes,'\n')
stopifnot(all(vapply(fits,`[[`,integer(1),'convergence')==0L), max(out$fit_values)-min(out$fit_values)<1e-7,max(abs(out$gradient))<1e-6,min(eig)>0.01,out$hessian_condition<1e4,out$gradient_fd_abs<1e-6,all(abs(theta)<4.9),max(abs(Hess-t(Hess)))<1e-6)
output <- commandArgs(trailingOnly = TRUE)
if (length(output) != 1L || file.exists(output)) stop('Supply a new output JSON path.')
jsonlite::write_json(out, output, digits=NA, pretty=TRUE, auto_unbox=TRUE)
cat(sprintf('PUBLIC_TREE_FIT_PASS n=%d theta=%s objective=%.14g score=%.3g min_eig=%.5g cond=%.5g max_start_objective_diff=%.3g fd_abs=%.3g\n',n,paste(format(theta,digits=10),collapse=','),f$value,max(abs(out$gradient)),min(eig),out$hessian_condition,max(out$fit_values)-min(out$fit_values),out$gradient_fd_abs))
