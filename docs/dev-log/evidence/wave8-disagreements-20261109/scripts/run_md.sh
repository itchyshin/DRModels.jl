source $HOME/disagree_w8_20261109/env.sh
cd $HOME/disagree_w8_20261109/harness
for id in "binomial|ordinary|lab0|q2|sd-one|cm0|ML" "binomial|ordinary|lab0|q2|sd-first|cm0|ML" "binomial|ordinary|lab0|q2|sd-z|cm0|ML" "binomial|ordinary|lab0|q2|sd-last|cm0|ML" "binomial|ordinary|lab0|q2|sd-none|cm0|ML" "poisson|ordinary|lab0|q2|sd-none|cm0|ML" "nb2|ordinary|lab0|q2|sd-none|cm0|ML" "binomial|indep|lab0|q2|sd-one|cm0|ML" "binomial|ordinary|lab0|q3|sd-first|cm0|ML" "binomial|ordinary|lab0|q4|sd-first|cm0|ML" "binomial|indep|lab0|q4|sd-z|cm0|ML"; do
  echo "$id"
done | xargs -P 11 -d '\n' -I{} bash -c 'nice -n 10 Rscript exact_md.R "{}" 2>&1 | grep -E "EXACT|Error" ' > ../out/exact_md.txt
echo MDDONE
