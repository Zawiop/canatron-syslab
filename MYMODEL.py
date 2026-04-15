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



def vp_reward(game, p0_color):
    winning_color = game.winning_color()
    if winning_color is None:
        return 0.0
    return 1.0 if winning_color == p0_color else -1.0

class RewardIteration2:
    def __init__(self):
        self.prev_vp = None
        self.prev_settlements = None
        self.prev_cities = None

    def _safe_get(self, obj, key, default=0):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _player_state(self, game, p0_color):
        ps = game.state.player_state
        if isinstance(ps, dict):
            if p0_color in ps:
                return ps[p0_color]
            for k, v in ps.items():
                if str(k) == str(p0_color):
                    return v
        return None

    def _count_structures(self, game, p0_color):
        settlements = 0
        cities = 0
        for _, b in game.state.board.buildings.items():
            color = self._safe_get(b, "color", None)
            btype = str(self._safe_get(b, "building_type", self._safe_get(b, "type", ""))).lower()
            if str(color) == str(p0_color):
                if btype.endswith("city"):
                    cities += 1
                else:
                    settlements += 1
        return settlements, cities

    def __call__(self, game, p0_color):
        winner = game.winning_color()
        if winner is not None:
            return 10.0 if str(winner) == str(p0_color) else -10.0

        player = self._player_state(game, p0_color)
        vp = self._safe_get(player, "victory_points",
             self._safe_get(player, "actual_victory_points",
             self._safe_get(player, "public_victory_points", 0)))

        settlements, cities = self._count_structures(game, p0_color)

        if self.prev_vp is None:
            self.prev_vp = vp
            self.prev_settlements = settlements
            self.prev_cities = cities
            return 0.0

        reward = 0.0
        reward += 2.5 * (vp - self.prev_vp)
        reward += 1.5 * (settlements - self.prev_settlements)
        reward += 2.5 * (cities - self.prev_cities)

        self.prev_vp = vp
        self.prev_settlements = settlements
        self.prev_cities = cities
        return float(reward)

import math



# -------------------------
# SINGLE ENV CREATOR
# -------------------------
reward = RewardIteration2()
def make_env():
    env = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "reward_function": reward,
            "invalid_action_reward": -1.0,
        },
    )
    env = ActionMasker(env, mask_fn)
    return env



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
 