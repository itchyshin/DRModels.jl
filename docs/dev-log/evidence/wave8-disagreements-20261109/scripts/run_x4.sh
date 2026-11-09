source $HOME/disagree_w8_20261109/env.sh
cd $HOME/disagree_w8_20261109
rm -f out/xe/*.tsv
cat ids_exact.tsv | xargs -P 22 -L 1 -d '\n' bash -c 'IFS=$'"'"'\t'"'"' read n id <<< "$0"; cd harness; N=100; case "$id" in tweedie*) N=60;; esac; nice -n 10 Rscript exact.R "$id" "$n" $N > ../out/xe_$n.log 2>&1'
echo ALLDONE4 $(date +%T)
