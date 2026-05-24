"""
Applied Project 3 - Imitation Learning
Step 3: Expert Dataset Generation

Rolls out π₁ and saves trajectories directly in SOAR-IL format:
    expert_data/states/{env_name}.pt    shape: (N, T, state_dim)  float32
    expert_data/actions/{env_name}.pt   shape: (N, T, action_dim) float32

Per-K files are also saved as {env_name}_K{k}.pt for explicit control.

Must be run from inside the SOAR-IL directory:
    cd SOAR-IL
    python ../step3_dataset_generation.py
    python ../step3_dataset_generation.py --env mcc
    python ../step3_dataset_generation.py --env mcc --inspect
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")

# ── Device ──
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

try:
    from stable_baselines3 import PPO, SAC, TD3
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# Output directories matching SOAR-IL layout exactly
STATES_DIR  = "expert_data/states"
ACTIONS_DIR = "expert_data/actions"
PLOTS_DIR   = "plots"

os.makedirs(STATES_DIR,  exist_ok=True)
os.makedirs(ACTIONS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

K_SIZES = [1, 5, 10, 20, 50]


# ─────────────────────────────────────────────────────────────────
# Expert policy loaders
# ─────────────────────────────────────────────────────────────────

def load_pendulum_expert(model_path="models/expert_pendulum"):
    if SB3_AVAILABLE and os.path.exists(model_path + ".zip"):
        model = SAC.load(model_path, device=DEVICE)
        print(f"  [Pendulum]  Loaded SAC model from disk (device={DEVICE}).")
        return lambda obs: model.predict(obs, deterministic=True)[0]
    print("  [Pendulum]  Using analytic PD expert (SAC model not found).")
    def policy(obs):
        cos_th, sin_th, thdot = obs
        th = np.arctan2(sin_th, cos_th)
        return np.clip(np.array([-3.0 * th - 0.5 * thdot]), -2.0, 2.0)
    return policy


def load_mcc_expert(model_path="models/expert_mcc"):
    if SB3_AVAILABLE and os.path.exists(model_path + ".zip"):
        model = TD3.load(model_path, device=DEVICE)
        print(f"  [MCC]       Loaded TD3 model from disk (device={DEVICE}).")
        return lambda obs: model.predict(obs, deterministic=True)[0]
    raise FileNotFoundError(
        f"TD3 model not found at {model_path}.zip. "
        "Please train the expert first."
    )


# ─────────────────────────────────────────────────────────────────
# Rollout  ← KEY FIX: final state is now correctly included
# ─────────────────────────────────────────────────────────────────

def rollout_trajectory(env, policy, seed=None):
    """
    Roll out one episode. Returns:
        states:  np.array (T, obs_dim)   — includes the terminal state
        actions: np.array (T-1, act_dim) — one action per transition
        total_return: float
    
    FIX vs original: we now append next_obs after the loop so the
    final state (where the goal is reached) is included in the trajectory.
    This ensures max_position > 0.45 is captured for MountainCarContinuous.
    """
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    states, actions, rewards = [], [], []

    while not done and not trunc:
        action = policy(obs)
        action_arr = np.atleast_1d(np.array(action, dtype=np.float32))
        next_obs, reward, done, trunc, _ = env.step(action)

        states.append(obs.copy())
        actions.append(action_arr)
        rewards.append(reward)

        obs = next_obs  # advance

    # ✅ Append the final observation (goal state for MCC)
    states.append(obs.copy())

    return (
        np.array(states,  dtype=np.float32),   # shape: (T+1, obs_dim)
        np.array(actions, dtype=np.float32),   # shape: (T,   act_dim)
        float(np.sum(rewards)),
    )


# ─────────────────────────────────────────────────────────────────
# Align states and actions to the same length T
# SOAR-IL expects states and actions tensors of equal T dimension.
# We trim states to match actions length (drop the extra terminal state
# for storage, but it was used above to verify goal was reached).
# ─────────────────────────────────────────────────────────────────

def align_and_pad(states, actions, T):
    """
    states:  (T_ep+1, obs_dim)  — includes terminal state
    actions: (T_ep,   act_dim)

    Pad actions with a zero action to match states length, then pad/trim both to T.
    This preserves the terminal state (goal position for MCC).
    """
    # Pad actions with a zero to match states length (T_ep+1)
    zero_action = np.zeros((1, actions.shape[1]), dtype=np.float32)
    actions = np.concatenate([actions, zero_action], axis=0)  # now (T_ep+1, act_dim)

    # Now both are (T_ep+1, *) — pad or trim to T
    t = states.shape[0]
    if t == T:
        return states, actions
    if t > T:
        return states[:T], actions[:T]
    # Pad with last frame
    s_pad = np.repeat(states[-1:],  T - t, axis=0)
    a_pad = np.repeat(actions[-1:], T - t, axis=0)
    return np.concatenate([states, s_pad]), np.concatenate([actions, a_pad])


# ─────────────────────────────────────────────────────────────────
# Save in SOAR-IL format
# ─────────────────────────────────────────────────────────────────

def save_soar_il(env_name, states_list, actions_list, k_tag=""):
    states_tensor  = torch.tensor(np.stack(states_list),  dtype=torch.float32)
    actions_tensor = torch.tensor(np.stack(actions_list), dtype=torch.float32)

    s_path = os.path.join(STATES_DIR,  f"{env_name}{k_tag}.pt")
    a_path = os.path.join(ACTIONS_DIR, f"{env_name}{k_tag}.pt")

    torch.save(states_tensor,  s_path)
    torch.save(actions_tensor, a_path)

    return s_path, a_path, tuple(states_tensor.shape), tuple(actions_tensor.shape)


# ─────────────────────────────────────────────────────────────────
# Verification: check trajectories are valid
# ─────────────────────────────────────────────────────────────────

def verify_trajectories(env_name, all_states, all_returns):
    """Print a sanity check on the saved trajectories."""
    print(f"\n  Verification for {env_name}:")
    print(f"    Mean return : {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    print(f"    Min / Max   : {np.min(all_returns):.2f} / {np.max(all_returns):.2f}")

    if "MountainCar" in env_name:
        # Each trajectory may have different length, so check per-trajectory
        max_pos_per_traj = np.array([s[:, 0].max() for s in all_states])
        reached = (max_pos_per_traj > 0.45).sum()
        print(f"    Max position: {max_pos_per_traj.mean():.4f} (need > 0.45)")
        print(f"    Reached goal: {reached}/{len(all_states)} trajectories")
        if reached == 0:
            print("    ⚠️  WARNING: No trajectory reached the goal!")
            print("       Your expert model may not be solving the environment.")
        else:
            print(f"    ✅ Expert is valid.")

    if "Pendulum" in env_name:
        good = (np.array(all_returns) > -300).sum()
        print(f"    Returns > -300: {good}/{len(all_returns)}")
        if good < len(all_returns) * 0.8:
            print("    ⚠️  WARNING: Many trajectories have poor returns.")
        else:
            print(f"    ✅ Expert is valid.")


# ─────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────

def generate_datasets(env_id, policy, k_sizes=K_SIZES, seed_offset=0):
    print(f"\n  Rolling out {max(k_sizes)} trajectories for {env_id} ...")

    env = gym.make(env_id)
    all_states_raw, all_actions_raw, all_returns = [], [], []

    for i in range(max(k_sizes)):
        s, a, ret = rollout_trajectory(env, policy, seed=seed_offset + i)
        all_states_raw.append(s)
        all_actions_raw.append(a)
        all_returns.append(ret)

        # For MCC: check if this trajectory reached the goal
        goal_info = ""
        if "MountainCar" in env_id:
            max_pos = s[:, 0].max()
            goal_info = f"  max_pos={max_pos:.3f} {'✅' if max_pos > 0.45 else '❌'}"

        print(f"    traj {i+1:>3}/{max(k_sizes)}  |"
              f"  steps={len(a):>4}  |  return={ret:8.2f}{goal_info}")

    env.close()

    # Verify before saving
    verify_trajectories(env_id, all_states_raw, all_returns)

    # Pad/trim all to the same length T (based on actions length)
    T = max(len(a) for a in all_actions_raw)
    print(f"\n  Padding/trimming all trajectories to T={T}")

    all_states_T  = []
    all_actions_T = []
    for s, a in zip(all_states_raw, all_actions_raw):
        s_aligned, a_aligned = align_and_pad(s, a, T)
        all_states_T.append(s_aligned)
        all_actions_T.append(a_aligned)

    # Save per-K files
    saved = []
    for K in k_sizes:
        s_path, a_path, s_shape, a_shape = save_soar_il(
            env_id, all_states_T[:K], all_actions_T[:K], k_tag=f"_K{K}"
        )
        mean_ret = float(np.mean(all_returns[:K]))
        saved.append((K, s_path, a_path, K * T, mean_ret))
        print(f"    K={K:>3}  states={s_shape}  actions={a_shape}"
              f"  mean_return={mean_ret:.2f}")

    # Default file = full K=max
    max_K = max(k_sizes)
    s_path, _, s_shape, _ = save_soar_il(
        env_id, all_states_T[:max_K], all_actions_T[:max_K], k_tag=""
    )
    print(f"\n  Default (K={max_K}): {s_path}  shape={s_shape}")
    print(f"  Control K at runtime via:  expert_episodes: <K>  in the YAML config")

    return all_returns, saved


# ─────────────────────────────────────────────────────────────────
# Inspection utility
# ─────────────────────────────────────────────────────────────────

def inspect_dataset(env_id, K):
    s_path = os.path.join(STATES_DIR,  f"{env_id}_K{K}.pt")
    a_path = os.path.join(ACTIONS_DIR, f"{env_id}_K{K}.pt")
    if not os.path.exists(s_path):
        print(f"  [SKIP] {s_path} not found")
        return
    s = torch.load(s_path, weights_only=True)
    a = torch.load(a_path, weights_only=True)
    print(f"\n  {env_id}  K={K}")
    print(f"    states  : {tuple(s.shape)}  dtype={s.dtype}")
    print(f"    actions : {tuple(a.shape)}  dtype={a.dtype}")
    if "MountainCar" in env_id:
        max_pos = s[:, :, 0].max().item()
        print(f"    max position reached: {max_pos:.4f}  "
              f"{'✅ goal reached' if max_pos > 0.45 else '❌ goal NOT reached'}")


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

def plot_dataset_stats(all_stats):
    n = len(all_stats)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    colors = {
        "Pendulum-v1":              "#E91E63",
        "MountainCarContinuous-v0": "#4CAF50",
    }

    for col, (env_id, (all_returns, saved)) in enumerate(all_stats.items()):
        color     = colors.get(env_id, "#9C27B0")
        env_label = env_id.replace("MountainCarContinuous", "MCC")

        ax_top = axes[0][col]
        ax_top.hist(all_returns, bins=15, color=color, alpha=0.75, edgecolor="white")
        ax_top.axvline(np.mean(all_returns), color="black", linestyle="--",
                       linewidth=1.5, label=f"Mean: {np.mean(all_returns):.1f}")
        ax_top.axvline(np.min(all_returns), color="red", linestyle=":",
                       linewidth=1.2, label=f"Min: {np.min(all_returns):.1f}")
        ax_top.set_title(f"{env_label}\nReturn Distribution ({len(all_returns)} trajs)",
                         fontsize=11, fontweight="bold")
        ax_top.set_xlabel("Episode Return")
        ax_top.set_ylabel("Count")
        ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.3)

        ax_bot = axes[1][col]
        ks        = [r[0] for r in saved]
        mean_rets = [r[4] for r in saved]
        n_trans   = [r[3] for r in saved]

        ax_bot.plot(ks, mean_rets, color=color, linewidth=2.5,
                    marker="o", markersize=7, label="Mean return")
        for K, mr in zip(ks, mean_rets):
            std = np.std(all_returns[:K])
            ax_bot.errorbar(K, mr, yerr=std, fmt="none",
                            ecolor=color, elinewidth=1.5, capsize=4)

        ax_bot.set_title(f"{env_label}\nMean Return vs Dataset Size K",
                         fontsize=11, fontweight="bold")
        ax_bot.set_xlabel("K  (number of expert trajectories)")
        ax_bot.set_ylabel("Mean Episode Return")
        ax_bot.set_xticks(ks)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(fontsize=9)
        for K, mr, nt in zip(ks, mean_rets, n_trans):
            ax_bot.annotate(f"{nt} trans", xy=(K, mr),
                            xytext=(0, 10), textcoords="offset points",
                            fontsize=7, ha="center", color="gray")

    plt.suptitle("Step 3: Expert Dataset Statistics", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "dataset_statistics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

ENV_CONFIG = {
    "pendulum": ("Pendulum-v1",              load_pendulum_expert),
    "mcc":      ("MountainCarContinuous-v0", load_mcc_expert),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["pendulum", "mcc", "all"],
                        default="all")
    parser.add_argument("--k_sizes", nargs="+", type=int, default=K_SIZES,
                        help="Dataset sizes to generate, e.g. --k_sizes 1 5 20 50")
    parser.add_argument("--inspect", action="store_true",
                        help="Print summary of saved datasets after generation")
    args = parser.parse_args()

    envs_to_run = list(ENV_CONFIG.keys()) if args.env == "all" else [args.env]

    print("=" * 60)
    print("  STEP 3: Expert Dataset Generation (SOAR-IL format)")
    print("=" * 60)
    print(f"  Device       : {DEVICE}")
    print(f"  Environments : {envs_to_run}")
    print(f"  K sizes      : {args.k_sizes}")
    print(f"  Output dirs  : {STATES_DIR}/  {ACTIONS_DIR}/")
    print(f"  Run from     : {os.getcwd()}")

    all_stats = {}

    for env_key in envs_to_run:
        env_id, loader_fn = ENV_CONFIG[env_key]
        print(f"\n{'─'*60}")
        print(f"  {env_id}")
        print(f"{'─'*60}")
        policy = loader_fn()
        returns, saved = generate_datasets(env_id, policy, k_sizes=args.k_sizes)
        all_stats[env_id] = (returns, saved)

    plot_dataset_stats(all_stats)

    if args.inspect:
        print("\n" + "=" * 60)
        print("  Dataset Inspection")
        print("=" * 60)
        for env_key in envs_to_run:
            env_id, _ = ENV_CONFIG[env_key]
            for K in args.k_sizes:
                inspect_dataset(env_id, K)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_files = 0
    for env_id, (returns, saved) in all_stats.items():
        print(f"\n  {env_id}")
        print(f"    Expert mean return : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        for K, s_path, a_path, n_trans, mean_ret in saved:
            print(f"    K={K:>3}  {s_path}  ({n_trans} transitions, return={mean_ret:.2f})")
            total_files += 2
    print(f"\n  {total_files} .pt files saved")
    print("\n  To use in SOAR-IL:")
    print("    Set  expert_episodes: <K>  in the config YAML")
    print("    The repo will slice the default .pt file to the first K trajs.")
    print("=" * 60)