# train_catan_rl.py
import gymnasium as gym
import numpy as np
import catanatron.gym  # registers "catanatron/Catanatron-v0"

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env) -> np.ndarray:
    n = env.action_space.n
    mask = np.zeros(n, dtype=np.bool_)
    valid = env.unwrapped.get_valid_actions()
    valid = np.asarray(valid, dtype=np.int64).reshape(-1)
    valid = valid[(0 <= valid) & (valid < n)]

    if valid.size == 0:
        mask[:] = True
    else:
        mask[valid] = True

    return mask

def vp_reward(game, p0_color):
    winning_color = game.winning_color()
    if winning_color is None:
        return 0.0
    elif winning_color == p0_color:
        return 1.0
    else:
        return -1.0
    
def vp_progress_reward(game, p0_color):
    
    p0 = game.state.players[p0_color]

    
    vp = float(getattr(p0, "victory_points", 0))

    winning_color = game.winning_color()
    if winning_color is not None:
        return 10.0 if winning_color == p0_color else -10.0

    return 0.5 * vp


env = gym.make(
    "catanatron/Catanatron-v0",
    config={
        "reward_function": vp_reward,
        "invalid_action_reward": -1.0,
    },
)

env = ActionMasker(env, mask_fn)

model = MaskablePPO("MlpPolicy", env, verbose=1)

import time

hours = 6
TRAIN_SECONDS = 60 * 60 * hours
SAVE_EVERY = 60 * 10    

start = time.time()
last_save = start

while time.time() - start < TRAIN_SECONDS:
    model.learn(10_000, reset_num_timesteps=False)

    if time.time() - last_save > SAVE_EVERY:
        model.save("MYMODEL_autosave")
        last_save = time.time()

model.save("MYMODEL")



