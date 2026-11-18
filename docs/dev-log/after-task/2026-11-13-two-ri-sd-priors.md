# Two random-intercept SD priors

## Scope

Two distinct independent mean random-intercept groups in Gaussian ML can each have an optional Gamma prior on the natural SD. Fixed-effect priors may be used at the same time. The objective adds each normalized SD penalty once and does not add a log-SD Jacobian. Unsupported families and effect structures remain outside this slice.

## Validation and limits

The independent dense-Gaussian oracle exercises equal and asymmetric generating SDs, objective values, gradients and Hessians. The Julia tests compare the penalized objective and derivatives to frozen oracle records and exercise target ordering and inference refusals. They also check that bridge v2 is unchanged and v3 exposes both SD coordinates. See the focused test output and source pins in the PR validation record.

Live R-to-Julia bridge tests subsequently passed on Julia `4cf237b9e98d2314016447bc47a32b803e07ada7` and R `2a84bc0bbb4d2b81c14280e53efaa09fde620e16`, covering v1, v2 and v3, mixed priors and replay. These tests compare fitted objectives, penalties and scores. A subsequent live mixed-prior fit on R `6b065b5a` and Julia `4ed1be27` compared the bridge-returned inverse penalized curvature with the independent oracle at one fitted point; the maximum scaled error was 2.11e-15, and an SD-axis swap gave error 0.0608. The bridge does not expose arbitrary-point Hessian evaluation; this probe does not establish it or calibrated uncertainty. The full Julia package suite remains pending. A focused pass is not a full regression result. Bootstrap and uncertainty-dependent inference remain refused for active prior fits. The method is penalized ML; it does not provide posterior samples.
