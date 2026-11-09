cd ~/drm_ep_20261109/ep
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 FUZZ_DATA=data JULIA_DEPOT_PATH=$HOME/drm_ep_20261109/depot PATH=$HOME/.juliaup/bin:$PATH
mkdir -p runA
for i in 0 1 2; do
  nice -n 10 Rscript ep_r.R shard$i.tsv runA/r_shard$i.tsv > runA/r_shard$i.log 2>&1 &
  nice -n 10 julia +1.10.12 --project=$HOME/drm_ep_20261109/jenv ep_j.jl shard$i.tsv runA/j_shard$i.tsv > runA/j_shard$i.log 2>&1 &
done
wait; echo FINISHED > runA/finished.txt
