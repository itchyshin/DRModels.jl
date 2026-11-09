cd ~/drm_ep_20261109/ep
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 FUZZ_DATA=data JULIA_DEPOT_PATH=$HOME/drm_ep_20261109/depot PATH=$HOME/.juliaup/bin:$PATH
mkdir -p runB
for i in 0 1 2; do
  nice -n 10 julia +1.10.12 --project=$HOME/drm_ep_20261109/jenv ep_j2.jl shard$i.tsv runB/j2_shard$i.tsv > runB/j2_shard$i.log 2>&1 &
done
nice -n 10 timeout 2400 Rscript ep_prof_r.R prof10.tsv runB/prof_r.tsv > runB/prof_r.log 2>&1 &
nice -n 10 timeout 2400 julia +1.10.12 --project=$HOME/drm_ep_20261109/jenv ep_prof_j.jl prof10.tsv runB/prof_j.tsv > runB/prof_j.log 2>&1 &
wait; echo FINISHED > runB/finished.txt
