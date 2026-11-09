cd ~/drm_ep_20261109/ep
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 FUZZ_DATA=data JULIA_DEPOT_PATH=$HOME/drm_ep_20261109/depot PATH=$HOME/.juliaup/bin:$PATH
for i in 0 1 2; do
  nice -n 10 julia +1.10.12 --project=$HOME/drm_ep_20261109/jenv ep_j2.jl shard$i.tsv runB/j2_shard$i.tsv > runB/j2_shard$i.log 2>&1 &
done
wait; echo FINISHED > runB/finished2.txt
