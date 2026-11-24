source $HOME/disagree_w8_20261109/env.sh
cd $HOME/disagree_w8_20261109
echo phase2 $(date +%T)
cut -f1,2 ids_numbered.tsv | xargs -P 20 -L 1 -d '\n' bash -c 'IFS=$'"'"'\t'"'"' read n id <<< "$0"; nice -n 10 julia +1.10 --project=$JENV xj.jl "$id" "$n" > out/xj/$n.log 2>&1'
echo phase2done $(date +%T)
# exact quadrature for one-dimensional random-effect cells
grep -E '^[0-9]+	(poisson|binomial|nb2|gamma|beta|student|tweedie)\|(ordinary|slope)\|lab0\|q1\|' ids_numbered.tsv > ids_exact.tsv
wc -l ids_exact.tsv
cat ids_exact.tsv | xargs -P 22 -L 1 -d '\n' bash -c 'IFS=$'"'"'\t'"'"' read n id <<< "$0"; cd harness; N=100; case "$id" in tweedie*) N=60;; esac; nice -n 10 Rscript exact.R "$id" "$n" $N > ../out/xe_$n.log 2>&1'
echo ALLDONE3 $(date +%T)
