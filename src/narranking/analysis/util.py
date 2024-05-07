import os

import pandas as pd
import pytrec_eval as pe
from matplotlib import pyplot as plt

from narraplay.documentranking.config import DIAGRAMS_DIR

MEASURES = {
    'P_10': 'P@10',
    'P_20': 'P@20',
    'ndcg_cut_10': 'nDCG@10',
    'ndcg_cut_20': 'nDCG@20',
    'recall_1000': 'Recall@1000',
    'set_recall': 'Set@Recall'
}

PLT_DEFAULT_COLOR_CYCLE = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
PLT_VARIANT_COLOR_CYCLE = PLT_DEFAULT_COLOR_CYCLE.copy()
PLT_VARIANT_COLOR_CYCLE[1], PLT_VARIANT_COLOR_CYCLE[2] = PLT_VARIANT_COLOR_CYCLE[2], PLT_VARIANT_COLOR_CYCLE[1]

PLT_DEFAULT_HATCH_CYCLE = '\\ /XO*xo'
PLT_VARIANT_HATCH_CYCLE = '\\/XO*xo '


def extract_run(run: dict, rel_topics: set, measure: str):
    run = sorted(run.items(), key=lambda x: int(x[0]))
    indices = [k for k, _ in run]
    values = [v[measure] for _, v in run]

    run = {i: v for i, v in zip(indices, values)}
    for tid in rel_topics:
        if tid not in run:
            run[tid] = 0.0

    return run


def draw_bar_chart(dfs: list, subdir: str, measure: str, *args: str, color_cycle=None):
    if color_cycle is None:
        color_cycle = PLT_DEFAULT_COLOR_CYCLE
        hatch_cycle = PLT_DEFAULT_HATCH_CYCLE
    else:
        hatch_cycle = PLT_VARIANT_HATCH_CYCLE
    file_name = "_".join([*args, measure])
    file_path = os.path.join(DIAGRAMS_DIR, subdir)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_name = os.path.join(file_path, file_name)

    measure = MEASURES[measure]
    df = pd.concat(dfs, axis=1)
    df.index = pd.to_numeric(df.index)
    df.sort_index(inplace=True)
    fig = plt.figure()
    ax = df.plot.bar(xlabel=f'Topic', ylabel=measure, figsize=(20, 6), width=0.85, edgecolor='black', color=color_cycle)

    # plot grid
    ax.grid(color='#bcbcbc')
    ax.set_axisbelow(True)

    # bar hatches
    bars = ax.patches
    hatches = ''.join(h * len(df) for h in hatch_cycle)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # font size
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    # find the highest score or 1.1 for [0 to 1] scores
    max_value = max(1.1, max(row for fs in df for row in df[fs]))
    plt.ylim([0, max_value])
    plt.legend(fontsize=16)
    plt.savefig(os.path.join(DIAGRAMS_DIR, subdir, f"{file_name}.pdf"), format="pdf")
    plt.close(fig)


def load_qrel_from_file(path: str):
    with open(path, 'r') as file:
        return pe.parse_qrel(file)


def load_run_from_file(path: str):
    with open(path, 'r') as file:
        return pe.parse_run(file)
