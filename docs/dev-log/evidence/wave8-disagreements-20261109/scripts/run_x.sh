source $HOME/disagree_w8_20261109/env.sh
cd $HOME/disagree_w8_20261109
mkdir -p out/xr out/xj
nl -ba harness/ids54.txt | awk '{print $1"\t"$2}' > ids_numbered.tsv
echo phase1 $(date +%T)
cat ids_numbered.tsv | (cd harness; while IFS=$'\t' read n id; do echo "$n|$id"; done) > /dev/null
export -f 2>/dev/null
cut -f1,2 ids_numbered.tsv | xargs -P 12 -L 1 -d '\n' bash -c 'IFS=$'"'"'\t'"'"' read n id <<< "$0"; cd harness; nice -n 10 Rscript xr1.R "$id" "$n" > /dev/null 2>&1' 
echo phase1 done $(ls out/xr | wc -l) $(date +%T)
cut -f1,2 ids_numbered.tsv | xargs -P 20 -L 1 -d '\n' bash -c 'IFS=$'"'"'\t'"'"' read n id <<< "$0"; nice -n 10 julia +1.10 --project=$JENV xj.jl "$id" "$n" > out/xj/$n.log 2>&1'
echo phase2 done $(ls out/xj/*.tsv | wc -l) $(date +%T)
cut -f1,2 ids_numbered.tsv | xargs -P 12 -L 1 -d '\n' bash -c 'IFS=$'"'"'\t'"'"' read n id <<< "$0"; cd harness; nice -n 10 Rscript xr2.R "$id" "$n" > ../out/xr/$n.log2 2>&1'
echo ALLDONE $(date +%T)
