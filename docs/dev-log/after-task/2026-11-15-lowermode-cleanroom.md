# Independent lower-mode reference script

This note helps reviewers trace the independently written R reference script
[`agree_independent.R`](../lowermode-check-20261108/agree_independent.R). It
uses drmTMB's public interface to fit the lower-mode check's specified models.
The author used the existing experiment README, Julia data generator, recorded
tables and public API documentation as the contract. The previous R generator
and its GPL counterpart were not read or copied during this rewrite. This note
is a provenance and bounded reproduction record; it does not clear a release.

The script takes a generator output directory (the parent containing `data/`),
a tab-separated jobs file and a new output path. The R package, generated data
and recorded comparison tables remain external inputs. The script refuses an
existing output path. Its SHA-256 for the rerun was
`9c912849319e7f71937432aa754678ccb85772642c9367f6975bb01b30203dd2`.

## Rerun and limits

On Totoro, seed 1 of Gamma phylogenetic q1, Gamma spatial q1, NB2 phylogenetic
q2, NB2 spatial q2 and Gamma phylogenetic q2 reproduced the recorded R log
likelihoods within 4.3e-11. The four original routes also matched their recorded
optimizer, Hessian and convergence flags. The q2 Gamma follow-up table does not
record those R flags. This comparison checks five selected cases; it does not
rerun the other seeds, establish Julia parity or prove numerical correctness.

The rerun used R 4.6.1 and drmTMB 0.8.1 loaded from
`/home/snakagaw/drm_campaign_20261108/Rlib/drmTMB`, with OpenBLAS, OpenMP and
MKL thread settings each set to one. The installed DESCRIPTION SHA-256 was
`7560d95c74647a194dea09784c71c4b0778bccd0fdec0f12e748df9c48f34c00` and its
compiled library SHA-256 was
`1a1628c92d00cc25446999d262e179e6ac0da9016630f20569570ed7868d6c46`.
The installed build's source commit is unknown: the version and fingerprints
identify the observed installation but do not establish a source revision.

The command, environment, jobs, output, complete log, comparison, status and
before/after input hashes are retained under
`/Users/z3437171/.codex/evidence/unified-prior-table-20261114/cleanroom-lowermode-rerun/`.
The input hash comparison was empty, and the run, comparison and available flag
checks exited successfully. The output SHA-256 was
`2cc7743a94172969754a7bbc4947284f56da1c7c9927018484b34577e2d299a2`.
