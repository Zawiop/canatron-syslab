# train_catan_rl.py
import gymnasium as gym
import numpy as np
import catanatron.gym  # registers "catanatron/Catanatron-v0"

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env) -> np.ndarray:
    # Catanatron exposes the currently valid discrete actions as ints
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[env.unwrapped.get_valid_actions()] = True
    return mask

env = gym.make("catanatron/Catanatron-v0")
env = ActionMasker(env, mask_fn)  # enable action masking

model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=400_000)
model.save("catan_maskableppo")
