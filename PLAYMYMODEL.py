# play_catan_rl.py
import gymnasium as gym
import numpy as np
import catanatron.gym

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

def mask_fn(env) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[env.unwrapped.get_valid_actions()] = True
    return mask

env = ActionMasker(gym.make("catanatron/Catanatron-v0"), mask_fn)
model = MaskablePPO.load("MYMODEL")

obs, info = env.reset()
while True:
    action_masks = get_action_masks(env)
    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
