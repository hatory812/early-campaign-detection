"""
スナップショット縮小グラフのサイズレポート.

train_twitter_snapshot.py の縮小規則に従い, 各時間幅 t_w において各グラフ
G_i を「拡散開始 (= 最小タイムスタンプ t_0) から t_w 分以内」のエッジ・ノードに
縮小したときの

    |V_ie| (= 生き残るノード数) と |E_ie| (= 生き残るエッジ数)

を数え, t_w ごとに全グラフ (および campaign / noncampaign 別) の統計値
(合計・平均・標準偏差・最小・各分位点・最大) をまとめたレポートを出力する.

訓練は行わない (サイズ集計のみ) ため高速で, JSON のパースは一度だけ行う.
グラフ選別 (exceptions / ラベル有無) とノードの位置付けは
train_twitter_snapshot.build_cache をそのまま流用しており, 学習スクリプトと
同一のデータセットに対する集計になる.

出力 (results/ 配下, base = snapshot_size_{rww}_nodeattr{na}):
  - {base}_detail.csv : (t_w, graph) ごとの |V_ie| / |E_ie| と全体に対する割合
  - {base}_summary.csv: t_w ごとの統計値 (overall / campaign / noncampaign)
  - {base}.json       : 上記すべてを含む生データ
  - {base}.md         : 人が読むためのレポート (代表 t_w の表)
  - {base}_VE.png     : |V_ie| / |E_ie| の平均 (min-max 帯付き) vs t_w
"""

import os
import json
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_twitter_snapshot as snap


def snapshot_sizes(c, t_w):
    """キャッシュ c を t_w 分でフィルタしたときの (|V_ie|, |E_ie|) を返す.

    build_snapshot と同じマスク (ts <= t0 + t_w*60) を使うが, Data を作らず
    数だけ数える.
    """
    threshold = c['t0'] + t_w * 60
    mask = c['ts'] <= threshold
    num_edges = int(mask.sum())
    if num_edges == 0:
        # build_snapshot 同様, ts == t0 のエッジが必ず残るので通常起きない.
        return 0, 0
    surviving = np.unique(np.concatenate([c['src'][mask], c['dst'][mask]]))
    return int(surviving.shape[0]), num_edges


def describe(values):
    """配列の統計量を dict で返す (空なら全て 0)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        keys = ['count', 'sum', 'mean', 'std', 'min', 'p25', 'median', 'p75', 'max']
        return {k: 0 for k in keys}
    return {
        'count': int(arr.size),
        'sum': int(arr.sum()),
        'mean': float(round(arr.mean(), 3)),
        'std': float(round(arr.std(), 3)),
        'min': int(arr.min()),
        'p25': float(round(np.percentile(arr, 25), 3)),
        'median': float(round(np.median(arr), 3)),
        'p75': float(round(np.percentile(arr, 75), 3)),
        'max': int(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="各 t_w における各 G_i の |V_ie| / |E_ie| とその統計レポートを出力する.")
    parser.add_argument('--data_type', default='all', help="Either small or all")
    parser.add_argument("--small_graphs_path", help="Mention path to small graphs")
    parser.add_argument("--all_graphs_path", help="Mention path to all graphs")
    parser.add_argument("--rww_attr", default="degree", help="Mention what feature for rww")
    parser.add_argument("--node_attr", default="1", help="Whether node features should be used")
    parser.add_argument("--classify_news", default=0, help="Classify the news graphs")
    parser.add_argument("--tw_min", default=1, type=int, help="Minimum t_w (minutes)")
    parser.add_argument("--tw_max", default=60, type=int, help="Maximum t_w (minutes)")
    parser.add_argument("--tw_step", default=1, type=int, help="Step for t_w sweep")
    args = parser.parse_args()

    rww_attr = args.rww_attr
    node_attr = int(args.node_attr)
    # build_cache が参照するグローバルを設定 (学習スクリプトと同一のグラフ選別にする).
    snap.classify_news = int(args.classify_news)

    data_path = args.all_graphs_path if args.data_type != 'small' else args.small_graphs_path

    cache = snap.build_cache(data_path, rww_attr, node_attr)
    print(f"Cache built: {len(cache)} graphs")

    # フルグラフ (= t_w が十分大きいときと等価) のサイズ. 割合計算の分母に使う.
    full_V = np.array([c['node_feat'].shape[0] for c in cache], dtype=np.float64)
    full_E = np.array([c['src'].shape[0] for c in cache], dtype=np.float64)
    labels = np.array([c['label'] for c in cache], dtype=np.int64)
    names = [c['name'] for c in cache]

    tw_values = list(range(args.tw_min, args.tw_max + 1, args.tw_step))

    detail_rows = []      # (t_w, graph) ごとの行
    summary_per_tw = []   # t_w ごとの統計

    for t_w in tw_values:
        Vs = np.empty(len(cache), dtype=np.int64)
        Es = np.empty(len(cache), dtype=np.int64)
        for i, c in enumerate(cache):
            v, e = snapshot_sizes(c, t_w)
            Vs[i] = v
            Es[i] = e

        for i in range(len(cache)):
            detail_rows.append({
                't_w': t_w, 'graph': names[i], 'label': int(labels[i]),
                'num_nodes': int(Vs[i]), 'num_edges': int(Es[i]),
                'full_nodes': int(full_V[i]), 'full_edges': int(full_E[i]),
                'frac_nodes': float(round(Vs[i] / full_V[i], 4)),
                'frac_edges': float(round(Es[i] / full_E[i], 4)),
            })

        camp = labels == 1
        noncamp = labels == 0
        stat = {
            't_w': t_w,
            'nodes': {
                'overall': describe(Vs),
                'campaign': describe(Vs[camp]),
                'noncampaign': describe(Vs[noncamp]),
            },
            'edges': {
                'overall': describe(Es),
                'campaign': describe(Es[camp]),
                'noncampaign': describe(Es[noncamp]),
            },
            'frac_nodes_mean': float(round((Vs / full_V).mean(), 4)),
            'frac_edges_mean': float(round((Es / full_E).mean(), 4)),
        }
        summary_per_tw.append(stat)

        print(f"t_w={t_w:2d}min | "
              f"|V|: mean={stat['nodes']['overall']['mean']:.1f} "
              f"min={stat['nodes']['overall']['min']} max={stat['nodes']['overall']['max']} | "
              f"|E|: mean={stat['edges']['overall']['mean']:.1f} "
              f"min={stat['edges']['overall']['min']} max={stat['edges']['overall']['max']} | "
              f"frac_E={stat['frac_edges_mean']:.3f}")

    out_base = os.path.join('results', f"snapshot_size_{rww_attr}_nodeattr{node_attr}")
    save_reports(out_base, args, names, labels, full_V, full_E,
                 detail_rows, summary_per_tw, tw_values)


def save_reports(out_base, args, names, labels, full_V, full_E,
                 detail_rows, summary_per_tw, tw_values):
    os.makedirs('results', exist_ok=True)

    # --- JSON (全データ) ---
    report = {
        'args': vars(args),
        'num_graphs': len(names),
        'num_campaign': int((labels == 1).sum()),
        'num_noncampaign': int((labels == 0).sum()),
        'full_graph': {
            'nodes': describe(full_V),
            'edges': describe(full_E),
        },
        'per_tw': summary_per_tw,
        'detail': detail_rows,
    }
    with open(out_base + '.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # --- detail CSV ---
    detail_fields = ['t_w', 'graph', 'label', 'num_nodes', 'num_edges',
                     'full_nodes', 'full_edges', 'frac_nodes', 'frac_edges']
    with open(out_base + '_detail.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    # --- summary CSV ---
    groups = ['overall', 'campaign', 'noncampaign']
    stat_keys = ['count', 'sum', 'mean', 'std', 'min', 'p25', 'median', 'p75', 'max']
    with open(out_base + '_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['t_w']
        for kind in ['V', 'E']:
            for g in groups:
                for sk in stat_keys:
                    header.append(f'{kind}_{g}_{sk}')
        header += ['frac_nodes_mean', 'frac_edges_mean']
        writer.writerow(header)
        for s in summary_per_tw:
            row = [s['t_w']]
            for kind, name in [('V', 'nodes'), ('E', 'edges')]:
                for g in groups:
                    for sk in stat_keys:
                        row.append(s[name][g][sk])
            row += [s['frac_nodes_mean'], s['frac_edges_mean']]
            writer.writerow(row)

    # --- Markdown レポート ---
    write_markdown(out_base + '.md', report, summary_per_tw, tw_values)

    # --- PNG (|V| / |E| 平均 + min-max 帯) ---
    try:
        plot_sizes(out_base + '_VE.png', summary_per_tw, args.rww_attr)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print(f"Saved reports to {out_base}.{{json,md,_detail.csv,_summary.csv,_VE.png}}")


def write_markdown(path, report, summary_per_tw, tw_values):
    fg = report['full_graph']
    lines = []
    lines.append("# スナップショット縮小グラフ サイズレポート")
    lines.append("")
    lines.append(f"- 対象グラフ数: **{report['num_graphs']}** "
                 f"(campaign={report['num_campaign']}, "
                 f"noncampaign={report['num_noncampaign']})")
    lines.append(f"- rww_attr: `{report['args']['rww_attr']}`, "
                 f"node_attr: `{report['args']['node_attr']}`")
    lines.append(f"- t_w 範囲: {report['args']['tw_min']}..{report['args']['tw_max']} "
                 f"(step {report['args']['tw_step']}) 分")
    lines.append(f"- 縮小規則: `E_ie = {{ts <= t0 + t_w*60}}`, "
                 f"`V_ie = E_ie の両端ノード`")
    lines.append("")
    lines.append("## フルグラフ (縮小なし) のサイズ")
    lines.append("")
    lines.append("| | count | mean | std | min | median | max | sum |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(f"| ノード \\|V\\| | {fg['nodes']['count']} | {fg['nodes']['mean']} | "
                 f"{fg['nodes']['std']} | {fg['nodes']['min']} | {fg['nodes']['median']} | "
                 f"{fg['nodes']['max']} | {fg['nodes']['sum']} |")
    lines.append(f"| エッジ \\|E\\| | {fg['edges']['count']} | {fg['edges']['mean']} | "
                 f"{fg['edges']['std']} | {fg['edges']['min']} | {fg['edges']['median']} | "
                 f"{fg['edges']['max']} | {fg['edges']['sum']} |")
    lines.append("")
    lines.append("## t_w ごとの統計 (overall)")
    lines.append("")
    lines.append("|t_w (分)|\\|V\\| mean|\\|V\\| std|\\|V\\| min|\\|V\\| max|"
                 "\\|E\\| mean|\\|E\\| std|\\|E\\| min|\\|E\\| max|"
                 "frac V|frac E|")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary_per_tw:
        v = s['nodes']['overall']
        e = s['edges']['overall']
        lines.append(f"|{s['t_w']}|{v['mean']}|{v['std']}|{v['min']}|{v['max']}|"
                     f"{e['mean']}|{e['std']}|{e['min']}|{e['max']}|"
                     f"{s['frac_nodes_mean']}|{s['frac_edges_mean']}|")
    lines.append("")
    lines.append("> 各 (t_w, グラフ) ごとの個別の値は `*_detail.csv`, "
                 "campaign / noncampaign 別の統計は `*_summary.csv` を参照.")
    lines.append("")
    with open(path, 'w') as f:
        f.write("\n".join(lines))


def plot_sizes(path, summary_per_tw, rww_attr):
    tws = [s['t_w'] for s in summary_per_tw]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind, name in [(axes[0], 'V', 'nodes'), (axes[1], 'E', 'edges')]:
        mean = np.array([s[name]['overall']['mean'] for s in summary_per_tw])
        vmin = np.array([s[name]['overall']['min'] for s in summary_per_tw])
        vmax = np.array([s[name]['overall']['max'] for s in summary_per_tw])
        ax.plot(tws, mean, marker='.', color='C0', label='mean')
        ax.fill_between(tws, vmin, vmax, color='C0', alpha=0.15, label='min-max')
        ax.set_xlabel('t_w (minutes)')
        ax.set_ylabel(f'|{kind}| ({name})')
        ax.set_title(f'|{kind}| vs snapshot window ({rww_attr})')
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
