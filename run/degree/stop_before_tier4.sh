#!/usr/bin/env bash
# gnn03 の前計算を、段階 3 の完了直後 (段階 4 の作業プロセスが処理を始める前) に止める見張り役。
# 段階 4 は gnn04・gnn05 で実行中のため、gnn03 が同じ t_w を二重に処理しないようにする。
# ファイルの削除・移動はしない。DRY_RUN=1 なら停止せず対象のプロセスを表示するだけ。
L=/hss01/A.hattori/log_revised; LOG=$L/stop_watcher.log; RUN=/home/A.hattori/ECMLPKDD25_run_revised
MARK='--- Tier3 large  DONE ---'
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
find_main() { for p in $(pgrep -f 'precompute_minimized_embeddings[.]py'); do [ "$(readlink /proc/$p/cwd)" = "$RUN" ] && echo $p; done; }
desc() { local c; for c in $(pgrep -P "$1"); do echo $c; desc $c; done; }

MAIN=$(find_main)
[ -z "$MAIN" ] && { log "主プロセスが見つからないので終了 ($(hostname))"; exit 1; }
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN: 主プロセス $MAIN / 子孫 $(desc $MAIN | tr '\n' ' ') / 目印の行 $(grep -c -- "$MARK" $L/precompute_main.log) 件"; exit 0
fi
log "監視開始 ($(hostname)): 主プロセス $MAIN、目印「$MARK」を 2 秒おきに確認"
while ! grep -q -- "$MARK" $L/precompute_main.log; do
  kill -0 $MAIN 2>/dev/null || { log "段階 3 完了前に主プロセスが終了したので何もしない"; exit 0; }
  sleep 2
done
log "段階 3 完了を検知: $(grep -- "$MARK" $L/precompute_main.log | tail -1 | cut -c1-19)"
kill -STOP $MAIN                       # 主プロセスを凍結 (作業プロセスの再起動を防ぐ)
D=$(desc $MAIN | tr '\n' ' ')
log "主プロセス $MAIN を一時停止。子孫プロセス: ${D:-なし}"
[ -n "$D" ] && kill -KILL $D 2>/dev/null
kill -KILL $MAIN 2>/dev/null
sleep 3
left=""; for p in $MAIN $D; do kill -0 $p 2>/dev/null && left="$left $p"; done
log "停止完了。残っているプロセス: ${left:-なし}"
for t in $(seq 960 60 1440); do printf "%s:%s " $t $(ls /hss01/A.hattori/rww_all_graphs_minimized_revised/${t}min | wc -l); done | sed 's/^/段階 4 のファイル数 (gnn04・gnn05 の分を含む): /' | while read -r l; do log "$l"; done
