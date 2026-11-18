# After-task: phylogenetic fixed priors

This isolated slice adds normal penalties on mean and log residual-SD coefficients to a Gaussian model with one phylogenetic mean intercept and a phylogenetic SD regression. It retains the existing random-effect structure and stores the data likelihood separately from the penalized objective. The prior does not target coefficients in the phylogenetic SD regression.

The focused tests compare the engine with an independently reviewed dense likelihood and an interior 16-tip fit. The latter fixture was constructed after preliminary package comparisons. A live R-to-Julia test checks table targets and detects a changed target. These checks do not establish posterior inference, frequentist interval coverage, or reliability at every boundary.

The full package checks, source reconciliation, broader bridge validation and independent final review remain open. Missing responses, additional structured effects, the sparse prior route and other families are outside this first slice. No release or merge is implied by this record.
