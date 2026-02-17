import gymnasium as gym
import numpy as np
import torch
import catanatron.gym

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


# -------------------------
# MASK
# -------------------------
def mask_fn(env) -> np.ndarray:
    n = env.action_space.n
    valid = np.array(env.unwrapped.get_valid_actions(), dtype=np.int64).flatten()

    mask = np.zeros(n, dtype=np.bool_)
    valid = valid[(valid >= 0) & (valid < n)]

    if valid.size == 0:
        mask[0] = True
    else:
        mask[valid] = True
    return mask


# -------------------------
# REWARD
# -------------------------
def vp_reward(game, p0_color):
    winning_color = game.winning_color()
    if winning_color is None:
        return 0.0
    return 1.0 if winning_color == p0_color else -1.0


# -------------------------
# SINGLE ENV CREATOR
# -------------------------
def make_env():
    env = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "reward_function": vp_reward,
            "invalid_action_reward": -1.0,
        },
    )
    env = ActionMasker(env, mask_fn)
    return env


# -------------------------
# REQUIRED FOR WINDOWS
# -------------------------
if __name__ == "__main__":

    print("CUDA available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    N_ENVS =8

    # IMPORTANT: do NOT call make_env()
    venv = SubprocVecEnv([make_env for _ in range(N_ENVS)])

    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        device=device,      # GPU
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.05,
        n_steps=1024,
        batch_size=512,
        gamma=0.99,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=1000000,
        save_path="./checkpoints/",
        name_prefix="catan",
    )
    hours = 4
    timesteps = 5000000*hours
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_cb)
    finally:
        model.save(f"MYMODEL_final{timesteps}")
        venv.save("vec_normalize.pkl")
        print("Training finished + saved.")
 