#!/usr/bin/env Rscript
# Clean-room refits of existing CSV data. No data-generating algorithm is implied.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: Rscript refit-fixture.R CASE DATA.csv OUTPUT_DIR")
case <- args[[1L]]; input <- normalizePath(args[[2L]], mustWork = TRUE); out <- args[[3L]]
valid <- c("cumlogit-intercept", "cumlogit-slope-only", "tweedie-intercept", "tweedie-slope-only")
if (!case %in% valid) stop("Unknown case; choose: ", paste(valid, collapse = ", "))
if (dir.exists(out) && length(list.files(out, all.files=TRUE, no..=TRUE))) stop("Output directory must be empty")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
if (!requireNamespace("drmTMB", quietly=TRUE)) stop("Install drmTMB before running")
if (!requireNamespace("jsonlite", quietly=TRUE)) stop("Install jsonlite before running")
suppressPackageStartupMessages(library(drmTMB))
d <- utils::read.csv(input, stringsAsFactors=FALSE, check.names=FALSE)
if (!all(c("id", "x") %in% names(d)) || anyNA(d$id) || anyNA(d$x) || !is.numeric(d$x)) stop("Invalid id or x")
d$id <- factor(d$id)
if (case %in% c("tweedie-intercept", "tweedie-slope-only")) {
  if (!"y" %in% names(d) || !is.numeric(d$y) || any(!is.finite(d$y)) || any(d$y < 0)) stop("Invalid Tweedie response")
  if (case == "tweedie-intercept") {
    formula <- bf(y ~ x + (1 | id), sigma ~ 1, nu ~ 1)
    model_description <- "y ~ x + (1 | id); sigma ~ 1; nu ~ 1; Tweedie"
  } else {
    formula <- bf(y ~ x + (0 + x | id), sigma ~ 1, nu ~ 1)
    model_description <- "y ~ x + (0 + x | id); sigma ~ 1; nu ~ 1; Tweedie"
  }
  family <- tweedie()
} else {
  if (!"y_int" %in% names(d) || anyNA(d$y_int) || any(d$y_int != as.integer(d$y_int))) stop("Invalid ordinal response")
  categories <- sort(unique(as.integer(d$y_int)))
  if (!identical(categories, seq_len(length(categories)))) stop("Ordinal categories must be consecutive positive integers")
  d$y_int <- ordered(d$y_int, levels=categories)
  if (case == "cumlogit-intercept") {
    formula <- bf(y_int ~ x + (1 | id))
    model_description <- "y_int ~ x + (1 | id); cumulative logit, ordered categories"
  } else {
    formula <- bf(y_int ~ x + (0 + x | id))
    model_description <- "y_int ~ x + (0 + x | id); cumulative logit, ordered categories"
  }
  family <- cumulative_logit()
}
set.seed(20261115)
warnings <- character()
fit <- withCallingHandlers(drmTMB(formula, family=family, data=d, engine="tmb", REML=FALSE),
  warning=function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") })
coefs <- coef(fit)
coef_table <- do.call(rbind, lapply(names(coefs), function(nm) {
  v <- coefs[[nm]]
  data.frame(block=nm, term=names(v), estimate=unname(as.numeric(v)), row.names=NULL)
}))
utils::write.csv(coef_table, file.path(out,"coefficients.csv"),row.names=FALSE,na="")
saveRDS(fit,file.path(out,"fit.rds"),version=3)
summary_fit <- summary(fit)
utils::write.csv(summary_fit$parameters, file.path(out,"derived-parameters.csv"), row.names=FALSE, na="")
if (!is.null(summary_fit$ordinal$cutpoints)) {
  cp <- summary_fit$ordinal$cutpoints
  utils::write.csv(data.frame(cutpoint=names(cp), estimate=unname(cp)),
                   file.path(out,"cutpoints.csv"), row.names=FALSE, na="")
}
writeLines(capture.output(print(summary_fit)),file.path(out,"summary.txt"))
writeLines(capture.output(print(ranef(fit))),file.path(out,"random-effects.txt"))
writeLines(capture.output(sessionInfo()),file.path(out,"session.txt"))
sha256 <- function(path) {
  val <- system2("shasum", c("-a", "256", shQuote(path)), stdout=TRUE, stderr=TRUE)
  if (!is.null(attr(val,"status"))) stop("Cannot hash ",path)
  strsplit(val[[1L]], "[[:space:]]+")[[1L]][[1L]]
}
dll <- getLoadedDLLs()[["drmTMB"]][["path"]]
historical_ll <- c("cumlogit-intercept"=-1026.092, "cumlogit-slope-only"=-775.3768,
  "tweedie-intercept"=-572.2594, "tweedie-slope-only"=-591.8245)
result <- list(case=case, model=model_description, model_source="maintainer-supplied Julia-side public model contract",
  input_file=input, input_sha256=sha256(input), n=nrow(d), groups=nlevels(d$id),
  package_version=as.character(utils::packageVersion("drmTMB")),
  tmb_version=as.character(utils::packageVersion("TMB")), R_version=R.version.string,
  drmTMB_dll_path=dll, drmTMB_dll_sha256=sha256(dll), seed=20261115L,
  logLik=as.numeric(logLik(fit)), historical_logLik_approx=unname(historical_ll[[case]]),
  historical_source="Julia-side test values relayed by maintainer; drmTMB 0.7.0",
  difference_from_approx=as.numeric(logLik(fit))-unname(historical_ll[[case]]), convergence=tryCatch(is_converged(fit),error=function(e) NA),
  warnings=warnings, reference_comparison="approximate historical logLik only; no full expected-output artifact supplied")
jsonlite::write_json(result,file.path(out,"provenance.json"),auto_unbox=TRUE,pretty=TRUE,na="null",digits=17)
cat(jsonlite::toJSON(result,auto_unbox=TRUE,pretty=TRUE,na="null",digits=17),"\n")
