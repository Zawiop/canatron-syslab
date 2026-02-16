import gymnasium as gym
import numpy as np
import catanatron.gym
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1. Improved Mask Function with Safety Fallback
def mask_fn(env) -> np.ndarray:
    n = env.action_space.n
    # Ensure we get a numpy array of valid action indices
    valid = np.array(env.unwrapped.get_valid_actions(), dtype=np.int64).flatten()

    mask = np.zeros(n, dtype=np.bool_)
    
    # Filter valid actions to stay within bounds
    valid = valid[(valid >= 0) & (valid < n)]
    
    if valid.size == 0:
        # If the engine provides no valid actions, we must force one 
        # to prevent the "Simplex" distribution error. 
        # Usually, action 0 or the 'Pass' action is a safe bet.
        mask[0] = True 
    else:
        mask[valid] = True
    return mask

# 2. Reward Functions
def vp_reward(game, p0_color):
    winning_color = game.winning_color()
    if winning_color is None:
        return 0.0
    return 1.0 if winning_color == p0_color else -1.0

# 3. Environment Setup
def make_env():
    env = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "reward_function": vp_reward,
            "invalid_action_reward": -1.0,
        },
    )
    # Wrap with ActionMasker before vectorizing
    env = ActionMasker(env, mask_fn)
    return env

# Vectorize and Normalize
# This handles the "Exploding Gradients" that often lead to NaNs
venv = DummyVecEnv([make_env])
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

# 4. Model Definition with Conservative Hyperparameters
model = MaskablePPO(
    "MlpPolicy",
    venv,
    verbose=1,
    learning_rate=1e-4,     # Slower learning prevents distribution collapse
    ent_coef=0.05,          # Higher entropy encourages wider exploration
    batch_size=128,         # Larger batches stabilize the policy update
    n_steps=2048,
    gamma=0.99,
)

# 5. Training
checkpoint_cb = CheckpointCallback(
    save_freq=1_000_000,          
    save_path="./checkpoints/",
    name_prefix="catan"
)

try:
    model.learn(total_timesteps=5_000_000, callback=checkpoint_cb)
finally:
    # Always save the final model and the normalization stats
    model.save("MYMODEL_final")
    venv.save("vec_normalize.pkl")
    print("Training session ended. Model and Stats saved.")