# Two random-intercept SD priors

## Scope

Two distinct independent mean random-intercept groups in Gaussian ML can each have an optional Gamma prior on the natural SD. Fixed-effect priors may be used at the same time. The objective adds each normalized SD penalty once and does not add a log-SD Jacobian. Unsupported families and effect structures remain outside this slice.

## Validation and limits

The independent dense-Gaussian oracle exercises equal and asymmetric generating SDs, objective values, gradients and Hessians. The Julia tests compare the penalized objective and derivatives to frozen oracle records and exercise target ordering and inference refusals. They also check that bridge v2 is unchanged and v3 exposes both SD coordinates. See the focused test output and source pins in the PR validation record.

The full package suite and live R-to-Julia bridge validation are pending. A focused pass is not a full regression result. Bootstrap and uncertainty-dependent inference remain refused for active prior fits. The method is penalized ML; it does not provide posterior samples.
