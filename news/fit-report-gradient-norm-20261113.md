- **The printed fit report now shows the final gradient norm when the optimiser did not converge.**
  A fit with `converged = false` gets one more line, `final gradient: max|grad| = 1.234e-05`, after the status line.
  It is the same number as `check_drm(fit).max_abs_grad`, so you can see how far from stationary the reported point is.
  Fits whose optimiser converged print exactly what they printed before, and no fit, flag or tolerance changes.
  On Julia 1.10 the Gamma, Beta, Student and LogNormal `(1 + x | g)` fits that go through the line-search retry can still report
  `converged = false` at an optimum whose log-likelihood agrees with Julia 1.13 to about 1e-9; the new line shows how small the gradient is there.
