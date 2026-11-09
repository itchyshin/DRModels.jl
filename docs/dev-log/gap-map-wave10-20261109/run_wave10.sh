cd ~/drm_gap10_20261109/harness
export JULIA_DEPOT_PATH=$HOME/drm_gap10_20261109/depot JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export FUZZ_JULIA_PROJECT=$HOME/drm_gap10_20261109/jenv PATH=$HOME/.juliaup/bin:$PATH
# juliaup channel pin
export JULIAUP_CHANNEL=1.10.12
nice -n 10 python3 driver.py julia 40 300 cells_run.tsv out/julia > out_driver.stdout 2>&1
echo FINISHED > out_finished.txt
