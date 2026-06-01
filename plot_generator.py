"""
plot_generator.py
=================
Generates report-style plots comparing IQ-Learn, ML-IRL, and Soar+ML-IRL
across environments (CartPoleContinuous-v0 and Pendulum-v1).

Reads data directly from zip archives in algorithm_csv_results/:
  cartpole_iq.zip   — IQ-Learn CartPole results
  cartpole.zip      — MaxEntIRL CartPole results
  pendulum_iq.zip   — IQ-Learn Pendulum results
  pendulum.zip      — MaxEntIRL Pendulum results

Output plots (saved to OUTPUT_DIR):
    1. normalized_performance_vs_k.png  — normalized score vs K (line, log-scale x)
    2. mean_return_bar_vs_k.png         — grouped bar chart of final return per K
    3. learning_curves_all.png          — grid of training curves
    4. {env}_mlirl_vs_soar_per_k.png    — ML-IRL vs Soar+ML-IRL per k

Usage:
    python plot_generator.py
"""

import io
import os
import re
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    'font.size':              11,
    'axes.titlesize':         13,
    'axes.labelsize':         12,
    'xtick.labelsize':        10,
    'ytick.labelsize':        10,
    'legend.fontsize':        10,
    'legend.title_fontsize':  10,
    'lines.linewidth':        2.0,
})

METHOD_STYLES = {
    'IQ-Learn':    {'color': '#2166AC', 'marker': 'o', 'ls': '-'},
    'ML-IRL':      {'color': '#4DAC26', 'marker': 'o', 'ls': '-'},
    'Soar+ML-IRL': {'color': '#D01C8B', 'marker': 'o', 'ls': '-'},
}

BAR_COLORS = {
    'IQ-Learn':    '#2166AC',
    'ML-IRL':      '#4DAC26',
    'Soar+ML-IRL': '#D01C8B',
}

K_COLORS = {
    1:  '#6D28D9',
    5:  '#0891B2',
    10: '#059669',
    20: '#D97706',
    50: '#DC2626',
}


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_iq_zip(zip_path: str) -> pd.DataFrame:
    """
    Load IQ-Learn progress CSVs from a zip archive.
    Expected path pattern inside zip: .../exp-{k}/iq_learn/seed{N}/progress.csv
    """
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith('progress.csv'):
                continue
            m = re.search(r'exp-(\d+)/iq_learn/seed(\d+)/progress\.csv', name)
            if not m:
                continue
            k    = int(m.group(1))
            seed = int(m.group(2))
            with zf.open(name) as f:
                df = pd.read_csv(f)
            df['k']    = k
            df['seed'] = seed
            records.append(df)
    if not records:
        raise ValueError(f"No IQ-Learn data found in {zip_path}")
    return pd.concat(records, ignore_index=True)


def load_maxentirl_zip(zip_path: str) -> pd.DataFrame:
    """
    Load MaxEntIRL progress CSVs from a zip archive.
    Expected path pattern: .../exp-{k}/maxentirl/...{q_type}_seed{N}.../progress.csv
    q1 → ML-IRL, q4 → Soar+ML-IRL
    """
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if '__MACOSX' in name or not name.endswith('progress.csv'):
                continue
            m = re.search(r'exp-(\d+)/maxentirl/[^/]*(q\d+)_seed(\d+)', name)
            if not m:
                continue
            k      = int(m.group(1))
            q_type = m.group(2)
            seed   = int(m.group(3))
            with zf.open(name) as f:
                content = f.read()
            if not content.strip():
                continue
            df = pd.read_csv(io.BytesIO(content))
            df['k']      = k
            df['q_type'] = q_type
            df['seed']   = seed
            records.append(df)
    if not records:
        raise ValueError(f"No MaxEntIRL data found in {zip_path}")
    return pd.concat(records, ignore_index=True)


# ── Helper: final-iteration mean and std per k ─────────────────────────────────

def final_mean_std(df: pd.DataFrame, col: str, k_values: list):
    last_itr = df['Itration'].max()
    final    = df[df['Itration'] == last_itr]
    means, stds = [], []
    for k in k_values:
        vals = final[final['k'] == k][col]
        means.append(vals.mean())
        stds.append(vals.std())
    return np.array(means), np.array(stds)


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', lw=0.4, alpha=0.5)


# ── Plot 1: Normalized final performance vs K ──────────────────────────────────

def plot_normalized_vs_k(envs: list, k_values: list, out_path: str):
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        r_min = env['random_return']
        r_max = env['expert_return']

        def norm(arr):
            return (arr - r_min) / (r_max - r_min)

        def norm_std(s):
            return s / abs(r_max - r_min)

        method_data = {
            'IQ-Learn':    env['iq_df'],
            'ML-IRL':      env['irl_df'][env['irl_df']['q_type'] == 'q1'],
            'Soar+ML-IRL': env['irl_df'][env['irl_df']['q_type'] == 'q4'],
        }

        for method_name, df_m in method_data.items():
            means, stds = final_mean_std(df_m, 'Real Det Return', k_values)
            mn = norm(means)
            sn = norm_std(stds)
            st = METHOD_STYLES[method_name]
            ax.plot(k_values, mn, color=st['color'], marker=st['marker'],
                    ls=st['ls'], label=method_name, zorder=3)
            ax.fill_between(k_values, mn - sn, mn + sn,
                            color=st['color'], alpha=0.15)

        ax.axhline(1.0, color='black', lw=1.5, ls='--', alpha=0.7,
                   label='Expert reference')
        ax.set_xscale('log')
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel('Expert trajectories K')
        ax.set_ylabel('Normalized score (expert = 1)')
        ax.set_title(env['name'])
        ax.legend(loc='lower right', framealpha=0.9)
        _style(ax)

    fig.suptitle('Normalized final performance vs expert dataset size',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ── Plot 2: Bar chart of mean return ± std per K ───────────────────────────────

def plot_bar_vs_k(envs: list, k_values: list, out_path: str):
    n_methods = 3
    w         = 0.22
    group_w   = w * n_methods
    offsets   = np.linspace(-(group_w / 2 - w / 2), group_w / 2 - w / 2, n_methods)
    x         = np.arange(len(k_values))

    fig, axes = plt.subplots(1, len(envs), figsize=(6.5 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        method_data = [
            ('IQ-Learn',    env['iq_df']),
            ('ML-IRL',      env['irl_df'][env['irl_df']['q_type'] == 'q1']),
            ('Soar+ML-IRL', env['irl_df'][env['irl_df']['q_type'] == 'q4']),
        ]

        for j, (method_name, df_m) in enumerate(method_data):
            means, stds = final_mean_std(df_m, 'Real Det Return', k_values)
            ax.bar(x + offsets[j], means, w, yerr=stds, capsize=3,
                   color=BAR_COLORS[method_name], alpha=0.85, label=method_name,
                   error_kw=dict(elinewidth=1.2, ecolor='#333'))

        ax.axhline(env['expert_return'], color='gray', lw=1.2, ls='--',
                   alpha=0.7, label='Expert')
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={k}' for k in k_values])
        ax.set_xlabel('Expert trajectories K')
        ax.set_ylabel('Mean Return')
        ax.set_title(env['name'])
        ax.legend(fontsize=9, framealpha=0.9)
        _style(ax)

    fig.suptitle('Mean return ± std over 3 seeds vs expert trajectories K',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ── Plot 3: Learning curves grid ──────────────────────────────────────────────

def plot_learning_curves(envs: list, k_values: list, out_path: str):
    method_labels = ['IQ-Learn', 'ML-IRL', 'Soar+ML-IRL']
    n_rows = len(envs)
    n_cols = len(method_labels)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             sharey='row')
    if n_rows == 1:
        axes = [axes]

    for row, env in enumerate(envs):
        panels = [
            ('IQ-Learn',    env['iq_df']),
            ('ML-IRL',      env['irl_df'][env['irl_df']['q_type'] == 'q1']),
            ('Soar+ML-IRL', env['irl_df'][env['irl_df']['q_type'] == 'q4']),
        ]

        for col, (title, df_m) in enumerate(panels):
            ax = axes[row][col]
            for k in k_values:
                grp   = df_m[df_m['k'] == k].sort_values('Itration')
                pivot = grp.pivot_table(index='Itration', columns='seed',
                                        values='Real Det Return', aggfunc='mean')
                mean  = pivot.mean(axis=1)
                std   = pivot.std(axis=1)
                ax.plot(mean.index, mean.values, lw=1.8,
                        color=K_COLORS[k], label=f'k={k}')
                ax.fill_between(mean.index, mean - std, mean + std,
                                color=K_COLORS[k], alpha=0.15)

            ax.axhline(env['expert_return'], color='black', lw=1.2,
                       ls='--', alpha=0.6)
            if row == 0:
                ax.set_title(title, fontweight='semibold')
            if col == 0:
                ax.set_ylabel(f"{env['name']}\nDet Return", fontsize=10)
            ax.set_xlabel('Iteration')
            _style(ax)
            if row == 0 and col == n_cols - 1:
                ax.legend(title='k', fontsize=8, title_fontsize=9,
                          loc='lower right', framealpha=0.85)

    fig.suptitle('Deterministic Return over Training (mean ± std, 3 seeds)',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ── Plot 4: ML-IRL vs Soar+ML-IRL per k (5-panel grid) ───────────────────────

def plot_mlirl_vs_soar_per_k(env: dict, k_values: list, out_path: str):
    METHOD_COLOR = {'q1': '#3A8DC7', 'q4': '#E07B20'}
    METHOD_LABEL = {'q1': 'ML-IRL',  'q4': 'Soar+ML-IRL'}

    n = len(k_values)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=True)

    for ax, k in zip(axes, k_values):
        for q in ['q1', 'q4']:
            grp   = env['irl_df'][(env['irl_df']['q_type'] == q) &
                                  (env['irl_df']['k'] == k)].sort_values('Itration')
            pivot = grp.pivot_table(index='Itration', columns='seed',
                                    values='Real Det Return', aggfunc='mean')
            mean  = pivot.mean(axis=1)
            std   = pivot.std(axis=1)
            ax.plot(mean.index, mean.values, lw=2.0,
                    color=METHOD_COLOR[q], label=METHOD_LABEL[q])
            ax.fill_between(mean.index, mean - std, mean + std,
                            color=METHOD_COLOR[q], alpha=0.18)

        ax.set_title(f'k = {k}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', lw=0.4, alpha=0.5, color='#cccccc')
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel('Deterministic Return', fontsize=10)
    axes[-1].legend(loc='lower right', fontsize=9, framealpha=0.9,
                    handlelength=2.0)

    fig.suptitle(
        f'[{env["name"]}]  ML-IRL vs Soar+ML-IRL — Deterministic Return per k',
        fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    BASE = 'algorithm_csv_results'

    ENVIRONMENTS = [
        {
            'name':          'CartPoleContinuous-v0',
            'iq_zip':        os.path.join(BASE, 'cartpole_iq.zip'),
            'irl_zip':       os.path.join(BASE, 'cartpole.zip'),
            'expert_return': 500.0,
            'random_return': 0.0,
        },
        {
            'name':          'Pendulum-v1',
            'iq_zip':        os.path.join(BASE, 'pendulum_iq.zip'),
            'irl_zip':       os.path.join(BASE, 'pendulum.zip'),
            'expert_return': -118.0,
            'random_return': -1600.0,
        },
    ]

    K_VALUES   = [1, 5, 10, 20, 50]
    OUTPUT_DIR = 'report_plots'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    envs = []
    for cfg in ENVIRONMENTS:
        print(f"Loading {cfg['name']}...")
        iq_df  = load_iq_zip(cfg['iq_zip'])
        irl_df = load_maxentirl_zip(cfg['irl_zip'])
        envs.append({
            'name':          cfg['name'],
            'iq_df':         iq_df,
            'irl_df':        irl_df,
            'expert_return': cfg['expert_return'],
            'random_return': cfg['random_return'],
        })

    plot_normalized_vs_k(
        envs, K_VALUES,
        os.path.join(OUTPUT_DIR, 'normalized_performance_vs_k.png'))

    plot_bar_vs_k(
        envs, K_VALUES,
        os.path.join(OUTPUT_DIR, 'mean_return_bar_vs_k.png'))

    plot_learning_curves(
        envs, K_VALUES,
        os.path.join(OUTPUT_DIR, 'learning_curves_all.png'))

    for env in envs:
        safe_name = env['name'].replace('/', '_')
        plot_mlirl_vs_soar_per_k(
            env, K_VALUES,
            os.path.join(OUTPUT_DIR, f'{safe_name}_mlirl_vs_soar_per_k.png'))

    print("\nAll plots saved to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
