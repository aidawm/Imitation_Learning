"""
Applied Project 3 - Imitation Learning
Step 1: Environment Verification
Step 2: Expert Policy Training (π₁)

Environments:
  • CartPole-v1              → PPO  (discrete, easy benchmark)
  • Pendulum-v1              → SAC  (continuous, standard IL benchmark)
  • MountainCarContinuous-v0 → TD3 + OU noise  (sparse reward, hard exploration)

Usage:
  python step1_2_expert_training.py                  # train all 3
  python step1_2_expert_training.py --env cartpole   # train one only
  python step1_2_expert_training.py --env pendulum
  python step1_2_expert_training.py --env mcc
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)
os.makedirs("logs",   exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Callback: logs eval return at fixed intervals during training
# ─────────────────────────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=2000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.rewards         = []
        self.stds            = []
        self.timesteps       = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_r, std_r = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True, warn=False
            )
            self.rewards.append(mean_r)
            self.stds.append(std_r)
            self.timesteps.append(self.num_timesteps)
            if self.verbose:
                print(f"    step {self.num_timesteps:>8,} | return {mean_r:8.2f} ± {std_r:.2f}")
        return True


# ─────────────────────────────────────────────────────────────────
# STEP 1: Verify all environments
# ─────────────────────────────────────────────────────────────────

def step1_verify_environments():
    print("=" * 60)
    print("  STEP 1: Environment Verification")
    print("=" * 60)

    envs = [
        ("CartPole-v1",              "Discrete",    "Balancing pole on cart"),
        ("MountainCar-v0",           "Discrete",    "Car climbing a hill"),
        ("MountainCarContinuous-v0", "Continuous",  "Continuous hill climbing"),
        ("Acrobot-v1",               "Discrete",    "Two-link robot swing-up"),
        ("Pendulum-v1",              "Continuous",  "Pendulum swing-up"),
    ]

    print(f"\n  {'Environment':<28} {'Type':<12} {'Obs':<10} {'Action':<20}")
    print("  " + "-" * 72)

    for name, atype, desc in envs:
        env = gym.make(name)
        obs, _ = env.reset(seed=0)
        if hasattr(env.action_space, 'n'):
            act_str = f"Discrete({env.action_space.n})"
        else:
            lo = env.action_space.low[0]
            hi = env.action_space.high[0]
            act_str = f"Box[{lo:.1f},{hi:.1f}]^{env.action_space.shape[0]}"
        print(f"  {name:<28} {atype:<12} {str(obs.shape):<10} {act_str:<20}  OK")
        env.close()

    print("\n  All 5 environments verified.")
    print("\n  Selected for this project:")
    print("    CartPole-v1              -> PPO   (discrete)")
    print("    Pendulum-v1              -> SAC   (continuous, dense reward)")
    print("    MountainCarContinuous-v0 -> TD3   (continuous, sparse reward)")


# ─────────────────────────────────────────────────────────────────
# STEP 2a: CartPole-v1 with PPO
# ─────────────────────────────────────────────────────────────────

def train_cartpole(total_timesteps=100_000):
    print("\n" + "=" * 60)
    print("  STEP 2a: CartPole-v1  (PPO)")
    print("=" * 60)
    print("  Obs:  [cart_pos, cart_vel, pole_angle, pole_vel]")
    print("  Act:  push left (0) or right (1)")
    print("  Goal: keep pole upright for 500 steps  ->  max return = 500")

    train_env = make_vec_env("CartPole-v1", n_envs=4, seed=0)  # 4 parallel envs
    eval_env  = gym.make("CartPole-v1")

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate = 3e-4,
        n_steps       = 512,        # steps per env per update
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,       # entropy bonus for exploration
        verbose       = 0,
        seed          = 42,
        device        = "mps",
    )

    cb = RewardLoggerCallback(eval_env, eval_freq=2000, n_eval_episodes=20)
    print(f"\n  Training for {total_timesteps:,} steps (4 envs in parallel)...")
    model.learn(total_timesteps=total_timesteps, callback=cb)

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True, warn=False)
    print(f"\n  Final expert return: {mean_r:.1f} +/- {std_r:.1f}  (optimal = 500)")

    model.save("models/expert_cartpole")
    print("  Saved -> models/expert_cartpole.zip")

    train_env.close(); eval_env.close()
    return cb.timesteps, cb.rewards, cb.stds, mean_r


# ─────────────────────────────────────────────────────────────────
# STEP 2b: Pendulum-v1 with SAC
# ─────────────────────────────────────────────────────────────────

def train_pendulum(total_timesteps=60_000):
    print("\n" + "=" * 60)
    print("  STEP 2b: Pendulum-v1  (SAC)")
    print("=" * 60)
    print("  Obs:  [cos(theta), sin(theta), theta_dot]")
    print("  Act:  torque in [-2, 2]")
    print("  Goal: swing up and balance  ->  good return > -200")

    train_env = gym.make("Pendulum-v1")
    eval_env  = gym.make("Pendulum-v1")

    model = SAC(
        "MlpPolicy", train_env,
        learning_rate  = 3e-4,
        buffer_size    = 100_000,
        batch_size     = 256,
        gamma          = 0.99,
        tau            = 0.005,
        ent_coef       = "auto",    # automatic entropy tuning (key SAC feature)
        learning_starts= 1000,
        verbose        = 0,
        seed           = 42,
        device         = "mps",
    )

    cb = RewardLoggerCallback(eval_env, eval_freq=2000, n_eval_episodes=20)
    print(f"\n  Training for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, callback=cb)

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True, warn=False)
    print(f"\n  Final expert return: {mean_r:.1f} +/- {std_r:.1f}  (good = > -200, optimal ~ -120)")

    model.save("models/expert_pendulum")
    print("  Saved -> models/expert_pendulum.zip")

    train_env.close(); eval_env.close()
    return cb.timesteps, cb.rewards, cb.stds, mean_r


# ─────────────────────────────────────────────────────────────────
# STEP 2c: MountainCarContinuous-v0 with TD3 + OU noise
#
# WHY NOT SAC?
#   MCC has a very sparse reward (+100 only on goal, -0.1*action^2 otherwise).
#   SAC's entropy maximisation makes it reluctant to commit to large actions,
#   so it often never reaches the goal and stays stuck.
#
# WHY TD3 + OU NOISE?
#   Ornstein-Uhlenbeck noise is temporally correlated -> produces smooth,
#   sustained pushes that help the car build enough momentum to escape the
#   valley. Once a successful trajectory is in the replay buffer, learning
#   takes off quickly.
# ─────────────────────────────────────────────────────────────────

def train_mcc(total_timesteps=300_000):
    print("\n" + "=" * 60)
    print("  STEP 2c: MountainCarContinuous-v0  (TD3 + OU noise)")
    print("=" * 60)
    print("  Obs:  [position, velocity]")
    print("  Act:  force in [-1, 1]")
    print("  Goal: reach flag at pos=0.45  ->  good return > 90")
    print("\n  NOTE: sparse reward makes this the hardest of the three.")
    print("  TD3 + Ornstein-Uhlenbeck noise is the standard solution.")

    train_env = gym.make("MountainCarContinuous-v0")
    eval_env  = gym.make("MountainCarContinuous-v0")

    n_actions = train_env.action_space.shape[0]

    # OU noise: temporally correlated, generates sustained pushes
    # sigma=0.5 is intentionally large for aggressive early exploration
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean  = np.zeros(n_actions),
        sigma = 0.5 * np.ones(n_actions),
        theta = 0.15,
    )

    model = TD3(
        "MlpPolicy", train_env,
        learning_rate  = 1e-3,
        buffer_size    = 200_000,
        batch_size     = 256,
        gamma          = 0.99,
        tau            = 0.005,
        action_noise   = action_noise,
        learning_starts= 1000,
        policy_delay   = 2,         # TD3: delayed policy updates reduce overestimation
        verbose        = 0,
        seed           = 42,
        policy_kwargs  = dict(net_arch=[400, 300]),  # larger net for harder task
        device         = "mps",
    )

    cb = RewardLoggerCallback(eval_env, eval_freq=5000, n_eval_episodes=20)
    print(f"\n  Training for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, callback=cb)

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True, warn=False)
    print(f"\n  Final expert return: {mean_r:.1f} +/- {std_r:.1f}  (good = > 90, max ~ 93)")

    model.save("models/expert_mcc")
    print("  Saved -> models/expert_mcc.zip")

    train_env.close(); eval_env.close()
    return cb.timesteps, cb.rewards, cb.stds, mean_r


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

def plot_all(results: dict):
    """
    results = {
      'cartpole': (timesteps, rewards, stds, final),
      'pendulum': (...),
      'mcc':      (...),
    }
    """
    print("\n  Generating training curve plots...")

    configs = {
        "cartpole": dict(
            title="CartPole-v1  (PPO)",
            color="#2196F3",
            ref_y=500, ref_label="Optimal (500)",
            ylabel="Mean Episode Return",
            ylim=(0, 550),
        ),
        "pendulum": dict(
            title="Pendulum-v1  (SAC)",
            color="#E91E63",
            ref_y=-200, ref_label="Acceptable (−200)",
            ylabel="Mean Episode Return",
            ylim=None,
        ),
        "mcc": dict(
            title="MountainCarContinuous-v0  (TD3 + OU)",
            color="#4CAF50",
            ref_y=90, ref_label="Good policy (90)",
            ylabel="Mean Episode Return",
            ylim=None,
        ),
    }

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("Step 2: Expert Policy Training Curves (pi_1)", fontsize=14, fontweight="bold")
    if n == 1:
        axes = [axes]

    for ax, (key, (steps, rewards, stds, final)) in zip(axes, results.items()):
        cfg = configs[key]
        rewards = np.array(rewards)
        stds    = np.array(stds)

        ax.plot(steps, rewards, color=cfg["color"], linewidth=2.5,
                marker="o", markersize=4, label="Training")
        ax.fill_between(steps, rewards - stds, rewards + stds,
                        alpha=0.2, color=cfg["color"], label="+/- 1 std")
        ax.axhline(cfg["ref_y"], color="green", linestyle="--",
                   linewidth=1.5, label=cfg["ref_label"])
        ax.axhline(final, color=cfg["color"], linestyle=":",
                   linewidth=1.2, label=f"Final: {final:.1f}")

        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Timesteps")
        ax.set_ylabel(cfg["ylabel"])
        if cfg["ylim"]:
            ax.set_ylim(cfg["ylim"])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k")
        )

    plt.tight_layout()
    path = "plots/expert_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "pendulum", "mcc", "all"],
                        default="all", help="Which env to train")
    parser.add_argument("--steps_cartpole", type=int, default=100_000)
    parser.add_argument("--steps_pendulum", type=int, default=60_000)
    parser.add_argument("--steps_mcc",      type=int, default=300_000)
    args = parser.parse_args()

    step1_verify_environments()

    results = {}

    if args.env in ("cartpole", "all"):
        results["cartpole"] = train_cartpole(args.steps_cartpole)

    if args.env in ("pendulum", "all"):
        results["pendulum"] = train_pendulum(args.steps_pendulum)

    if args.env in ("mcc", "all"):
        results["mcc"] = train_mcc(args.steps_mcc)

    if results:
        plot_all(results)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    labels = {"cartpole": "CartPole (PPO)", "pendulum": "Pendulum (SAC)", "mcc": "MCC (TD3+OU)"}
    for key, (_, _, _, final) in results.items():
        print(f"  {labels[key]:<25} -> final return {final:.1f}")
    print("\n  Models saved in ./models/")
    print("  Plot  saved in ./plots/expert_training_curves.png")
    print("\n  Ready for Step 3: Expert Dataset Generation")
    print("=" * 60)
