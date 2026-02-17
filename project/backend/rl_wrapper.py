import gymnasium as gym
import catanatron.gym

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from catanatron.players.player import Player


# Reuse YOUR mask function exactly as defined
def mask_fn(env):
    import numpy as np
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[env.unwrapped.get_valid_actions()] = True
    return mask


class RLBOTPlayer(Player):
    def __init__(self, name, model_path):
        super().__init__(name)

        # EXACT same environment style as PLAYMYMODEL.py
        self.env = ActionMasker(
            gym.make("catanatron/Catanatron-v0"),
            mask_fn
        )

        self.model = MaskablePPO.load(model_path)

        self.obs, _ = self.env.reset()

    def decide_move(self, game):
        action_masks = get_action_masks(self.env)

        action, _ = self.model.predict(
            self.obs,
            action_masks=action_masks,
            deterministic=True,
        )

        self.obs, reward, terminated, truncated, _ = self.env.step(action)

        if terminated or truncated:
            self.obs, _ = self.env.reset()

        # Map action index to Catanatron move
        move = game.state.legal_moves[action]
        return move