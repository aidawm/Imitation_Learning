"""
Applied Project 3 - Imitation Learning
Step 3: Expert Dataset Generation + IQ-Learn Conversion

Rolls out π₁ and saves trajectories in TWO formats:
  1. SOAR-IL format  (your original pipeline):
       expert_data/states/{env_name}_K{k}.pt    shape: (K, T, state_dim)
       expert_data/actions/{env_name}_K{k}.pt   shape: (K, T, action_dim)

  2. IQ-Learn format  (for train_iq.py):
       expert_demos/{env_name}_K{k}.pkl         dict with keys:
                                                  states, next_states, actions,
                                                  rewards, dones, lengths

Usage:
    cd SOAR-IL
    python ../step3_dataset_generation.py                  # pendulum + mcc
    python ../step3_dataset_generation.py --env pendulum   # pendulum only
    python ../step3_dataset_generation.py --env pendulum --inspect
"""

import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")

from envs import ContinuousCartPoleWrapper


# ── Device ────────────────────────────────────────────────────────────────────
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
    print("  [WARNING] stable_baselines3 not found. "
          "Falling back to analytic expert for Pendulum.")

# ── Output directories ─────────────────────────────────────────────────────────
STATES_DIR   = "expert_data/states"
ACTIONS_DIR  = "expert_data/actions"
DEMOS_DIR    = "expert_demos"          # IQ-Learn pkl files go here
PLOTS_DIR    = "plots"

for d in [STATES_DIR, ACTIONS_DIR, DEMOS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

K_SIZES = [1, 5, 10, 20, 50]


# ── Expert policy loaders ─────────────────────────────────────────────────────

def load_cartpole_expert(model_path="models/expert_cartpole"):
    """
    Load the SAC expert trained on ContinuousCartPoleWrapper.
    The policy returns a continuous action in [-1, 1]; the wrapper
    handles the discrete mapping internally during rollout.
    """
    if SB3_AVAILABLE and os.path.exists(model_path + ".zip"):
        model = SAC.load(model_path, device=DEVICE)
        print(f"  [CartPole]  Loaded SAC model from disk (device={DEVICE}).")
        return lambda obs: model.predict(obs, deterministic=True)[0]

    raise FileNotFoundError(
        f"SAC model not found at {model_path}.zip.\n"
        "Train it first:\n"
        "  python step1_2_expert_training.py --env cartpole_sac"
    )


def load_pendulum_expert(model_path="models/expert_pendulum"):
    """
    Load a pretrained SAC expert for Pendulum-v1.
    Falls back to an analytic PD controller if no model is found.
    The analytic policy is surprisingly good on Pendulum (returns ~ -200).
    """
    if SB3_AVAILABLE and os.path.exists(model_path + ".zip"):
        model = SAC.load(model_path, device=DEVICE)
        print(f"  [Pendulum]  Loaded SAC model from disk (device={DEVICE}).")
        return lambda obs: model.predict(obs, deterministic=True)[0]

    print("  [Pendulum]  SAC model not found — using analytic PD expert.")
    print("              To train a proper expert run:")
    print("                python train_expert.py --env pendulum")

    def pd_policy(obs):
        cos_th, sin_th, thdot = obs
        th = np.arctan2(sin_th, cos_th)
        torque = -3.0 * th - 0.5 * thdot
        return np.clip(np.array([torque], dtype=np.float32), -2.0, 2.0)

    return pd_policy


def load_mcc_expert(model_path="models/expert_mcc"):
    """Load TD3 expert for MountainCarContinuous-v0."""
    if SB3_AVAILABLE and os.path.exists(model_path + ".zip"):
        model = TD3.load(model_path, device=DEVICE)
        print(f"  [MCC]       Loaded TD3 model from disk (device={DEVICE}).")
        return lambda obs: model.predict(obs, deterministic=True)[0]

    raise FileNotFoundError(
        f"TD3 model not found at {model_path}.zip.\n"
        "Please train the MCC expert first:\n"
        "  python train_expert.py --env mcc"
    )


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout_trajectory(env, policy, seed=None):
    """
    Roll out one full episode.

    Returns
    -------
    states       : np.ndarray  (T+1, obs_dim)   includes terminal observation
    actions      : np.ndarray  (T,   act_dim)
    rewards      : np.ndarray  (T,)
    dones        : np.ndarray  (T,)              True only at last real step
    total_return : float
    """
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    states, actions, rewards, dones = [], [], [], []

    while not done and not trunc:
        action = policy(obs)
        action_arr = np.atleast_1d(np.array(action, dtype=np.float32))
        next_obs, reward, done, trunc, _ = env.step(action)

        states.append(obs.copy())
        actions.append(action_arr)
        rewards.append(float(reward))
        dones.append(bool(done or trunc))

        obs = next_obs

    # Append final observation so SOAR-IL can see the terminal state
    states.append(obs.copy())

    return (
        np.array(states,  dtype=np.float32),   # (T+1, obs_dim)
        np.array(actions, dtype=np.float32),   # (T,   act_dim)
        np.array(rewards, dtype=np.float32),   # (T,)
        np.array(dones,   dtype=bool),         # (T,)
        float(np.sum(rewards)),
    )


# ── SOAR-IL helpers ───────────────────────────────────────────────────────────

def align_and_pad(states, actions, T):
    """
    Pad actions with a zero so it matches states length, then pad/trim
    both arrays to length T for uniform tensor stacking.
    """
    zero_action = np.zeros((1, actions.shape[1]), dtype=np.float32)
    actions = np.concatenate([actions, zero_action], axis=0)   # (T_ep+1, act_dim)

    t = states.shape[0]
    if t == T:
        return states, actions
    if t > T:
        return states[:T], actions[:T]
    # Pad by repeating the last frame
    s_pad = np.repeat(states[-1:],  T - t, axis=0)
    a_pad = np.repeat(actions[-1:], T - t, axis=0)
    return np.concatenate([states, s_pad]), np.concatenate([actions, a_pad])


def save_soar_il(env_name, states_list, actions_list, k_tag=""):
    """Save padded tensors in SOAR-IL format."""
    states_tensor  = torch.tensor(np.stack(states_list),  dtype=torch.float32)
    actions_tensor = torch.tensor(np.stack(actions_list), dtype=torch.float32)

    s_path = os.path.join(STATES_DIR,  f"{env_name}{k_tag}.pt")
    a_path = os.path.join(ACTIONS_DIR, f"{env_name}{k_tag}.pt")

    torch.save(states_tensor,  s_path)
    torch.save(actions_tensor, a_path)

    return s_path, a_path, tuple(states_tensor.shape), tuple(actions_tensor.shape)


# ── IQ-Learn conversion ───────────────────────────────────────────────────────

def save_iqlearn(env_name, trajs, K, k_tag=""):
    """
    Save K trajectories in IQ-Learn's ExpertDataset format.

    IQ-Learn's ExpertDataset expects a dict with these keys:
        "states"      : list of K arrays, each shape (T_i, obs_dim)
        "next_states" : list of K arrays, each shape (T_i, obs_dim)
        "actions"     : list of K arrays, each shape (T_i, act_dim)
        "rewards"     : list of K arrays, each shape (T_i,)
        "dones"       : list of K arrays, each shape (T_i,)
        "lengths"     : np.ndarray of shape (K,) — number of steps per trajectory

    Parameters
    ----------
    trajs : list of (states, actions, rewards, dones) tuples
            states  shape: (T+1, obs_dim)  — includes terminal obs at index T
            actions shape: (T,   act_dim)
            rewards shape: (T,)
            dones   shape: (T,)
    """
    expert_dict = {
        "states":      [],
        "next_states": [],
        "actions":     [],
        "rewards":     [],
        "dones":       [],
        "lengths":     [],
    }

    for states, actions, rewards, dones in trajs[:K]:
        T = len(actions)  # real number of steps; states has T+1 entries

        # states[:-1] = observations at t=0..T-1
        # states[1:]  = next observations at t=0..T-1  (stored during rollout)
        expert_dict["states"].append(states[:-1])                    # (T, obs_dim)
        expert_dict["next_states"].append(states[1:])                # (T, obs_dim)
        expert_dict["actions"].append(actions)                       # (T, act_dim)
        expert_dict["rewards"].append(rewards)                       # (T,)
        expert_dict["dones"].append(dones.astype(np.float32))        # (T,)
        expert_dict["lengths"].append(T)

    expert_dict["lengths"] = np.array(expert_dict["lengths"])        # (K,)

    out_path = os.path.join(DEMOS_DIR, f"{env_name}{k_tag}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(expert_dict, f)

    n_trans = int(expert_dict["lengths"].sum())
    return out_path, n_trans


# ── Verification ──────────────────────────────────────────────────────────────

def verify_trajectories(env_name, all_states, all_returns):
    print(f"\n  Verification for {env_name}:")
    print(f"    Mean return : {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    print(f"    Min / Max   : {np.min(all_returns):.2f} / {np.max(all_returns):.2f}")

    if "CartPole" in env_name:
        good = (np.array(all_returns) >= 490).sum()
        print(f"    Returns >= 490 (near-optimal) : {good}/{len(all_returns)}")
        if good < len(all_returns) * 0.8:
            print("    WARNING: Many trajectories fall short of near-optimal.")
        else:
            print("    Expert is valid.")

    if "MountainCar" in env_name:
        max_pos_per_traj = np.array([s[:, 0].max() for s in all_states])
        reached = (max_pos_per_traj > 0.45).sum()
        print(f"    Max position reached : {max_pos_per_traj.mean():.4f} (need > 0.45)")
        print(f"    Reached goal         : {reached}/{len(all_states)} trajectories")
        if reached == 0:
            print("    ⚠️  WARNING: No trajectory reached the goal!")
        else:
            print("    ✅ Expert is valid.")

    if "Pendulum" in env_name:
        good = (np.array(all_returns) > -300).sum()
        print(f"    Returns > -300 : {good}/{len(all_returns)}")
        if good < len(all_returns) * 0.8:
            print("    ⚠️  WARNING: Many trajectories have poor returns.")
        else:
            print("    ✅ Expert is valid.")


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_datasets(env_id, policy, k_sizes=K_SIZES, seed_offset=0, env_factory=None):
    max_K = max(k_sizes)
    print(f"\n  Rolling out {max_K} trajectories for {env_id} ...")

    env = env_factory() if env_factory is not None else gym.make(env_id)

    # Storage for raw (unpadded) trajectories
    all_states_raw   = []
    all_actions_raw  = []
    all_rewards_raw  = []
    all_dones_raw    = []
    all_returns      = []

    for i in range(max_K):
        s, a, r, d, ret = rollout_trajectory(env, policy, seed=seed_offset + i)
        all_states_raw.append(s)
        all_actions_raw.append(a)
        all_rewards_raw.append(r)
        all_dones_raw.append(d)
        all_returns.append(ret)

        goal_info = ""
        if "MountainCar" in env_id:
            max_pos = s[:, 0].max()
            goal_info = f"  max_pos={max_pos:.3f} {'✅' if max_pos > 0.45 else '❌'}"

        print(f"    traj {i+1:>3}/{max_K}  |  steps={len(a):>4}"
              f"  |  return={ret:8.2f}{goal_info}")

    env.close()

    # ── Verify ──
    verify_trajectories(env_id, all_states_raw, all_returns)

    # ── SOAR-IL: pad all to uniform T and save ──
    T = max(len(a) for a in all_actions_raw)
    print(f"\n  [SOAR-IL] Padding all trajectories to T={T}")

    all_states_T  = []
    all_actions_T = []
    for s, a in zip(all_states_raw, all_actions_raw):
        s_al, a_al = align_and_pad(s, a, T)
        all_states_T.append(s_al)
        all_actions_T.append(a_al)

    soar_saved = []
    for K in k_sizes:
        s_path, a_path, s_shape, a_shape = save_soar_il(
            env_id, all_states_T[:K], all_actions_T[:K], k_tag=f"_K{K}"
        )
        mean_ret = float(np.mean(all_returns[:K]))
        soar_saved.append((K, s_path, a_path, K * T, mean_ret))
        print(f"    [SOAR-IL] K={K:>3}  states={s_shape}  actions={a_shape}"
              f"  mean_return={mean_ret:.2f}")

    # Default file = full K
    save_soar_il(env_id, all_states_T[:max_K], all_actions_T[:max_K], k_tag="")
    print(f"    [SOAR-IL] Default file saved (K={max_K})")

    # ── IQ-Learn: build trajectory dict and save ──
    print(f"\n  [IQ-Learn] Converting to ExpertDataset pkl format ...")

    # Bundle raw trajectories for the converter
    raw_trajs = list(zip(all_states_raw, all_actions_raw,
                         all_rewards_raw, all_dones_raw))

    iq_saved = []
    for K in k_sizes:
        pkl_path, n_trans = save_iqlearn(env_id, raw_trajs, K, k_tag=f"_K{K}")
        mean_ret = float(np.mean(all_returns[:K]))
        iq_saved.append((K, pkl_path, n_trans, mean_ret))
        print(f"    [IQ-Learn] K={K:>3}  transitions={n_trans:>6}"
              f"  mean_return={mean_ret:.2f}  -> {pkl_path}")

    # Default pkl = full K
    pkl_path_default, _ = save_iqlearn(env_id, raw_trajs, max_K, k_tag="")
    print(f"    [IQ-Learn] Default pkl saved -> {pkl_path_default}")

    return all_returns, soar_saved, iq_saved


# ── Inspection ────────────────────────────────────────────────────────────────

def inspect_soar(env_id, K):
    s_path = os.path.join(STATES_DIR,  f"{env_id}_K{K}.pt")
    a_path = os.path.join(ACTIONS_DIR, f"{env_id}_K{K}.pt")
    if not os.path.exists(s_path):
        print(f"  [SKIP] {s_path} not found")
        return
    s = torch.load(s_path, weights_only=True)
    a = torch.load(a_path, weights_only=True)
    print(f"\n  [SOAR-IL] {env_id}  K={K}")
    print(f"    states  : {tuple(s.shape)}  dtype={s.dtype}")
    print(f"    actions : {tuple(a.shape)}  dtype={a.dtype}")
    if "MountainCar" in env_id:
        max_pos = s[:, :, 0].max().item()
        print(f"    max position: {max_pos:.4f}  "
              f"{'✅ goal reached' if max_pos > 0.45 else '❌ NOT reached'}")


def inspect_iqlearn(env_id, K):
    pkl_path = os.path.join(DEMOS_DIR, f"{env_id}_K{K}.pkl")
    if not os.path.exists(pkl_path):
        print(f"  [SKIP] {pkl_path} not found")
        return
    with open(pkl_path, "rb") as f:
        expert_dict = pickle.load(f)

    # Validate expected keys
    expected_keys = {"states", "next_states", "actions", "rewards", "dones", "lengths"}
    assert expected_keys.issubset(expert_dict.keys()), \
        f"Missing keys: {expected_keys - expert_dict.keys()}"

    K_actual = len(expert_dict["lengths"])
    total_trans = int(expert_dict["lengths"].sum())

    print(f"\n  [IQ-Learn] {env_id}  K={K}")
    print(f"    Trajectories      : {K_actual}")
    print(f"    Total transitions : {total_trans}")
    print(f"    Lengths           : {expert_dict['lengths'].tolist()}")
    print(f"    states[0]  shape  : {expert_dict['states'][0].shape}")
    print(f"    actions[0] shape  : {expert_dict['actions'][0].shape}")
    print(f"    rewards[0] sample : {expert_dict['rewards'][0][:3]}")
    done_count = sum(d.sum() for d in expert_dict["dones"])
    print(f"    Done flags (total): {int(done_count)}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_dataset_stats(all_stats):
    n = len(all_stats)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    colors = {
        "CartPole-v1":              "#673AB7",
        "Pendulum-v1":              "#E91E63",
        "MountainCarContinuous-v0": "#4CAF50",
    }

    for col, (env_id, (all_returns, soar_saved, _)) in enumerate(all_stats.items()):
        color     = colors.get(env_id, "#9C27B0")
        env_label = env_id.replace("MountainCarContinuous", "MCC")

        # Top: return histogram
        ax_top = axes[0][col]
        ax_top.hist(all_returns, bins=15, color=color, alpha=0.75, edgecolor="white")
        ax_top.axvline(np.mean(all_returns), color="black", linestyle="--",
                       linewidth=1.5, label=f"Mean: {np.mean(all_returns):.1f}")
        ax_top.axvline(np.min(all_returns), color="red", linestyle=":",
                       linewidth=1.2, label=f"Min:  {np.min(all_returns):.1f}")
        ax_top.set_title(f"{env_label}\nReturn Distribution ({len(all_returns)} trajs)",
                         fontsize=11, fontweight="bold")
        ax_top.set_xlabel("Episode Return")
        ax_top.set_ylabel("Count")
        ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.3)

        # Bottom: mean return vs K
        ax_bot = axes[1][col]
        ks        = [r[0] for r in soar_saved]
        mean_rets = [r[4] for r in soar_saved]
        n_trans   = [r[3] for r in soar_saved]

        ax_bot.plot(ks, mean_rets, color=color, linewidth=2.5,
                    marker="o", markersize=7, label="Mean return")
        for K, mr in zip(ks, mean_rets):
            std = np.std(all_returns[:K])
            ax_bot.errorbar(K, mr, yerr=std, fmt="none",
                            ecolor=color, elinewidth=1.5, capsize=4)
        for K, mr, nt in zip(ks, mean_rets, n_trans):
            ax_bot.annotate(f"{nt} trans", xy=(K, mr),
                            xytext=(0, 10), textcoords="offset points",
                            fontsize=7, ha="center", color="gray")

        ax_bot.set_title(f"{env_label}\nMean Return vs Dataset Size K",
                         fontsize=11, fontweight="bold")
        ax_bot.set_xlabel("K  (number of expert trajectories)")
        ax_bot.set_ylabel("Mean Episode Return")
        ax_bot.set_xticks(ks)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(fontsize=9)

    plt.suptitle("Step 3: Expert Dataset Statistics", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "dataset_statistics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved -> {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _make_cartpole_env():
    return ContinuousCartPoleWrapper(gym.make("CartPole-v1"))


ENV_CONFIG = {
    "cartpole": ("CartPole-v1",              load_cartpole_expert, _make_cartpole_env),
    "pendulum": ("Pendulum-v1",              load_pendulum_expert, None),
    "mcc":      ("MountainCarContinuous-v0", load_mcc_expert,      None),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "pendulum", "mcc", "all"],
                        default="all", help="Which environment(s) to generate data for")
    parser.add_argument("--k_sizes", nargs="+", type=int, default=K_SIZES,
                        help="Dataset sizes, e.g. --k_sizes 1 5 20 50")
    parser.add_argument("--inspect", action="store_true",
                        help="Print a summary of saved files after generation")
    args = parser.parse_args()

    envs_to_run = list(ENV_CONFIG.keys()) if args.env == "all" else [args.env]

    print("=" * 65)
    print("  STEP 3: Expert Dataset Generation")
    print("  Outputs: SOAR-IL (.pt) + IQ-Learn (.pkl)")
    print("=" * 65)
    print(f"  Device       : {DEVICE}")
    print(f"  Environments : {envs_to_run}")
    print(f"  K sizes      : {args.k_sizes}")
    print(f"  SOAR-IL dir  : {STATES_DIR}/  {ACTIONS_DIR}/")
    print(f"  IQ-Learn dir : {DEMOS_DIR}/")

    all_stats = {}

    for env_key in envs_to_run:
        env_id, loader_fn, env_factory = ENV_CONFIG[env_key]
        print(f"\n{'─' * 65}")
        print(f"  {env_id}")
        print(f"{'─' * 65}")
        policy = loader_fn()
        returns, soar_saved, iq_saved = generate_datasets(
            env_id, policy, k_sizes=args.k_sizes, env_factory=env_factory
        )
        all_stats[env_id] = (returns, soar_saved, iq_saved)

    plot_dataset_stats(all_stats)

    if args.inspect:
        print("\n" + "=" * 65)
        print("  Dataset Inspection")
        print("=" * 65)
        for env_key in envs_to_run:
            env_id, _ = ENV_CONFIG[env_key]
            for K in args.k_sizes:
                inspect_soar(env_id, K)
                inspect_iqlearn(env_id, K)

    # ── Final summary ──
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    for env_id, (returns, soar_saved, iq_saved) in all_stats.items():
        print(f"\n  {env_id}")
        print(f"    Expert mean return : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"    SOAR-IL files:")
        for K, s_path, _, n_trans, mean_ret in soar_saved:
            print(f"      K={K:>3}  {s_path}  ({n_trans} transitions, mean={mean_ret:.2f})")
        print(f"    IQ-Learn files:")
        for K, pkl_path, n_trans, mean_ret in iq_saved:
            print(f"      K={K:>3}  {pkl_path}  ({n_trans} transitions, mean={mean_ret:.2f})")

    print("\n" + "=" * 65)
    print("  HOW TO USE WITH IQ-LEARN")
    print("=" * 65)
    print("\n  1. Copy pkl files into IQ-Learn's experts/ folder:")
    print("       cp expert_demos/Pendulum-v1_K*.pkl <path-to>/IQ-Learn/iq_learn/experts/")
    print("\n  2. Run IQ-Learn (from inside iq_learn/):")
    print("       WANDB_MODE=disabled python train_iq.py \\")
    print("         env=pendulum agent=sac method=iq \\")
    print("         expert.demos=<K> seed=0")
    print("=" * 65)