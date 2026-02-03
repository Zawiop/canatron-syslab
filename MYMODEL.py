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

env = gym.make(
    "catanatron/Catanatron-v0",
    config={
        "reward_function": vp_reward,
        "invalid_action_reward": -1.0,
    },
)

env = ActionMasker(env, mask_fn)

model = MaskablePPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100_000)
model.save("MYMODEL")
print(model.policy)


