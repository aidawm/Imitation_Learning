"""
Applied Project 3 - Imitation Learning
Step 3: Expert Dataset Generation

For each environment, rolls out the trained expert policy π₁ and saves
K trajectories as a dataset. Datasets of multiple sizes are saved so
that the IL algorithms (Step 5) can be evaluated against varying K.

Each trajectory is a dict:
    {
        'observations': np.ndarray  (T, obs_dim),
        'actions':      np.ndarray  (T, act_dim) or (T,) for discrete,
        'rewards':      np.ndarray  (T,),
        'next_observations': np.ndarray (T, obs_dim),
        'dones':        np.ndarray  (T,)   bool
    }

A dataset file (expert_<env>_K<k>.pkl) is a list of K such dicts.

Usage:
    python step3_dataset_generation.py            # generate all
    python step3_dataset_generation.py --env mcc  # one env only
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")

# Try importing SB3 for trained models; fall back to analytic experts
try:
    from stable_baselines3 import PPO, SAC, TD3
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

os.makedirs("datasets", exist_ok=True)
os.makedirs("plots",    exist_ok=True)

# K values to generate — cover sparse, moderate, and rich regimes
K_SIZES = [1, 5, 10, 20, 50]


# ─────────────────────────────────────────────────────────────────
# Expert policy wrappers
# Each returns a callable: obs -> action (np.ndarray)
# ─────────────────────────────────────────────────────────────────

def load_cartpole_expert(model_path="models/expert_cartpole"):
    """PPO expert. Falls back to near-optimal analytic policy."""
    if SB3_AVAILABLE and os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, device="mps")
        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
        print("  [CartPole]  Loaded PPO model from disk.")
    else:
        # Analytic: simple energy-based controller
        # CartPole: push in direction of pole lean + cart correction
        def policy(obs):
            cart_pos, cart_vel, pole_angle, pole_vel = obs
            # Linear combination tuned to be near-optimal
            action = 1 if (pole_angle + 0.5 * pole_vel + 0.05 * cart_pos + 0.1 * cart_vel) > 0 else 0
            return action
        print("  [CartPole]  Using analytic expert (PPO model not found).")
    return policy


def load_pendulum_expert(model_path="models/expert_pendulum"):
    """SAC expert. Falls back to PD controller (suboptimal but usable)."""
    if SB3_AVAILABLE and os.path.exists(model_path + ".zip"):
        model = SAC.load(model_path, device="mps")
        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
        print("  [Pendulum]  Loaded SAC model from disk.")
    else:
        # Analytic PD swing-up controller
        def policy(obs):
            cos_th, sin_th, thdot = obs
            th = np.arctan2(sin_th, cos_th)
            torque = np.clip([-3.0 * th - 0.5 * thdot], -2.0, 2.0)
            return torque
        print("  [Pendulum]  Using analytic PD expert (SAC model not found).")
    return policy


def load_mcc_expert(model_path="models/expert_mcc"):
    """TD3 expert. Falls back to energy-pumping controller (near-optimal)."""
    if SB3_AVAILABLE and os.path.exists(model_path + ".zip"):
        model = TD3.load(model_path, device="mps")
        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
        print("  [MCC]       Loaded TD3 model from disk.")
    else:
        # Energy-pumping: always push in direction of current velocity
        # This is near-optimal and gets return ~89 consistently
        def policy(obs):
            pos, vel = obs
            force = 1.0 if vel >= 0 else -1.0
            return np.array([force], dtype=np.float32)
        print("  [MCC]       Using energy-pump expert (TD3 model not found).")
    return policy


# ─────────────────────────────────────────────────────────────────
# Core rollout function
# ─────────────────────────────────────────────────────────────────

def rollout_trajectory(env, policy, seed=None):
    """
    Roll out policy for one episode.
    Returns a trajectory dict and the total episodic return.
    """
    obs, _ = env.reset(seed=seed)
    done = trunc = False

    observations      = []
    actions           = []
    rewards           = []
    next_observations = []
    dones             = []

    while not done and not trunc:
        action = policy(obs)

        # Ensure action is array for uniform storage
        action_stored = np.atleast_1d(np.array(action, dtype=np.float32))

        next_obs, reward, done, trunc, _ = env.step(action)

        observations.append(obs.copy())
        actions.append(action_stored)
        rewards.append(reward)
        next_observations.append(next_obs.copy())
        dones.append(done or trunc)

        obs = next_obs

    trajectory = {
        "observations":       np.array(observations,      dtype=np.float32),
        "actions":            np.array(actions,            dtype=np.float32),
        "rewards":            np.array(rewards,            dtype=np.float32),
        "next_observations":  np.array(next_observations, dtype=np.float32),
        "dones":              np.array(dones,              dtype=bool),
    }
    return trajectory, float(np.sum(rewards))


# ─────────────────────────────────────────────────────────────────
# Dataset generation for one environment
# ─────────────────────────────────────────────────────────────────

def generate_datasets(env_id, policy, k_sizes=K_SIZES, seed_offset=0):
    """
    Generate and save datasets of sizes k_sizes for env_id.
    Always generates max(k_sizes) trajectories, then subsets for smaller K.
    Returns per-trajectory returns and per-K metadata.
    """
    print(f"\n  Generating up to K={max(k_sizes)} trajectories for {env_id}...")

    env = gym.make(env_id)
    env_tag = env_id.replace("-", "_").replace("v0", "v0").replace("v1", "v1")

    all_trajectories = []
    all_returns      = []

    max_K = max(k_sizes)
    for i in range(max_K):
        traj, ret = rollout_trajectory(env, policy, seed=seed_offset + i)
        all_trajectories.append(traj)
        all_returns.append(ret)

        steps = len(traj["rewards"])
        print(f"    traj {i+1:>3}/{max_K}  |  steps={steps:>4}  |  return={ret:8.2f}")

    env.close()

    # Save subsets for each K
    saved_files = []
    for K in k_sizes:
        subset   = all_trajectories[:K]
        filename = f"datasets/expert_{env_tag}_K{K}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(subset, f)

        n_transitions = sum(len(t["rewards"]) for t in subset)
        mean_ret      = np.mean([np.sum(t["rewards"]) for t in subset])
        saved_files.append((K, filename, n_transitions, mean_ret))
        print(f"    Saved K={K:>3}: {filename}  "
              f"({n_transitions:>5} transitions, mean_return={mean_ret:.2f})")

    return all_returns, saved_files


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

def plot_dataset_stats(all_stats: dict):
    """
    all_stats = {
        env_id: (all_returns, saved_files)
    }
    Produces two plots:
      A) Return distribution per trajectory (histogram)
      B) Mean return vs K (dataset size effect)
    """
    print("\n  Generating dataset statistics plots...")

    n = len(all_stats)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    colors = {"CartPole-v1": "#2196F3",
              "Pendulum-v1": "#E91E63",
              "MountainCarContinuous-v0": "#4CAF50"}

    for col, (env_id, (all_returns, saved_files)) in enumerate(all_stats.items()):
        color = colors.get(env_id, "#9C27B0")
        env_label = env_id.replace("MountainCarContinuous", "MCC")

        # ── Top: return distribution across all trajectories ──
        ax_top = axes[0][col]
        ax_top.hist(all_returns, bins=15, color=color, alpha=0.75, edgecolor="white")
        ax_top.axvline(np.mean(all_returns), color="black", linestyle="--",
                       linewidth=1.5, label=f"Mean: {np.mean(all_returns):.1f}")
        ax_top.axvline(np.min(all_returns),  color="red",   linestyle=":",
                       linewidth=1.2, label=f"Min:  {np.min(all_returns):.1f}")
        ax_top.set_title(f"{env_label}\nExpert Return Distribution (K={len(all_returns)} trajs)",
                         fontsize=11, fontweight="bold")
        ax_top.set_xlabel("Episode Return")
        ax_top.set_ylabel("Count")
        ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.3)

        # ── Bottom: mean return vs K ──
        ax_bot = axes[1][col]
        ks          = [f[0] for f in saved_files]
        mean_rets   = [f[3] for f in saved_files]
        n_trans     = [f[2] for f in saved_files]

        ax_bot.plot(ks, mean_rets, color=color, linewidth=2.5,
                    marker="o", markersize=7, label="Mean return")

        # Shade +/- std computed from the growing subsets
        for K, mean_r in zip(ks, mean_rets):
            sub = all_returns[:K]
            std = np.std(sub)
            ax_bot.errorbar(K, mean_r, yerr=std, fmt="none",
                            ecolor=color, elinewidth=1.5, capsize=4)

        ax_bot.set_title(f"{env_label}\nMean Expert Return vs Dataset Size K",
                         fontsize=11, fontweight="bold")
        ax_bot.set_xlabel("K  (number of expert trajectories)")
        ax_bot.set_ylabel("Mean Episode Return")
        ax_bot.set_xticks(ks)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(fontsize=9)

        # Annotate transition counts
        for K, mean_r, n_t in zip(ks, mean_rets, n_trans):
            ax_bot.annotate(f"{n_t} trans", xy=(K, mean_r),
                            xytext=(0, 10), textcoords="offset points",
                            fontsize=7, ha="center", color="gray")

    plt.suptitle("Step 3: Expert Dataset Statistics", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = "plots/dataset_statistics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# Dataset inspection utility
# ─────────────────────────────────────────────────────────────────

def inspect_dataset(filepath):
    """Load a saved dataset and print a summary."""
    with open(filepath, "rb") as f:
        trajs = pickle.load(f)

    print(f"\n  Inspecting: {filepath}")
    print(f"    Trajectories : {len(trajs)}")
    print(f"    Obs shape    : {trajs[0]['observations'].shape}")
    print(f"    Act shape    : {trajs[0]['actions'].shape}")
    total_trans = sum(len(t['rewards']) for t in trajs)
    print(f"    Total transitions: {total_trans}")
    returns = [np.sum(t['rewards']) for t in trajs]
    print(f"    Return  mean={np.mean(returns):.2f}  "
          f"std={np.std(returns):.2f}  "
          f"min={np.min(returns):.2f}  "
          f"max={np.max(returns):.2f}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

ENV_CONFIG = {
    "cartpole": ("CartPole-v1",              load_cartpole_expert),
    "pendulum": ("Pendulum-v1",              load_pendulum_expert),
    "mcc":      ("MountainCarContinuous-v0", load_mcc_expert),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "pendulum", "mcc", "all"],
                        default="all")
    parser.add_argument("--k_sizes", nargs="+", type=int, default=K_SIZES,
                        help="Dataset sizes to generate, e.g. --k_sizes 1 5 20 50")
    parser.add_argument("--inspect", action="store_true",
                        help="Print summary of saved datasets after generation")
    args = parser.parse_args()

    envs_to_run = (
        list(ENV_CONFIG.keys()) if args.env == "all" else [args.env]
    )

    print("=" * 60)
    print("  STEP 3: Expert Dataset Generation")
    print("=" * 60)
    print(f"  Environments : {envs_to_run}")
    print(f"  K sizes      : {args.k_sizes}")
    print(f"  Output dir   : ./datasets/")

    all_stats = {}

    for env_key in envs_to_run:
        env_id, loader_fn = ENV_CONFIG[env_key]
        print(f"\n{'─'*60}")
        print(f"  {env_id}")
        print(f"{'─'*60}")

        policy = loader_fn()
        returns, saved_files = generate_datasets(
            env_id, policy, k_sizes=args.k_sizes
        )
        all_stats[env_id] = (returns, saved_files)

    # Plot statistics
    plot_dataset_stats(all_stats)

    # Optional inspection
    if args.inspect:
        print("\n" + "=" * 60)
        print("  Dataset Inspection")
        print("=" * 60)
        for env_key in envs_to_run:
            env_id, _ = ENV_CONFIG[env_key]
            env_tag = env_id.replace("-", "_")
            for K in args.k_sizes:
                path = f"datasets/expert_{env_tag}_K{K}.pkl"
                if os.path.exists(path):
                    inspect_dataset(path)

    # Final summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_files = 0
    for env_id, (returns, saved_files) in all_stats.items():
        env_label = env_id.replace("MountainCarContinuous", "MCC")
        print(f"\n  {env_label}")
        print(f"    Expert mean return : {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
        for K, fname, n_trans, mean_ret in saved_files:
            print(f"    K={K:>3}: {fname:<45} ({n_trans} transitions)")
            total_files += 1

    print(f"\n  {total_files} dataset files saved in ./datasets/")
    print("  Ready for Step 4/5: IL Algorithm Training")
    print("=" * 60)
