path="/hss01/A.hattori/all_graphs"
files=(${IDS:-$(seq 1 323)})
comp=("0.5" "mid" "median")
pick="degree"

for file in "${files[@]}";do
  for c in "${comp[@]}";do
    #echo $file
    python3 ../../kcore_rww.py --path $path --out_root "${OUT_ROOT:-/hss01/A.hattori/rww_all_graphs_revised}" --graphId $file --pick $pick --comp $c
  done
done
