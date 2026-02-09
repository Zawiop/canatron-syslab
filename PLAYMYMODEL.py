# ppo_player.py
import os
import numpy as np

from catanatron import Player
from catanatron.cli import register_cli_player

from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.catanatron_env import to_action_space  # path may vary by version
from sb3_contrib.ppo_mask import MaskablePPO

FEATURES = get_feature_ordering(num_players=2)

class PPOPlayer(Player):
    def __init__(self, color, model_path=None):
        super().__init__(color)
        # Allow overriding model path via env var or param
        model_path = model_path or os.getenv("CATAN_PPO_MODEL", "MYMODEL.zip")
        self.model = MaskablePPO.load(model_path)

    def decide(self, game, playable_actions):
        # 1) Encode observation exactly like training (vector rep)
        sample = create_sample(game, self.color)
        obs = np.array([float(sample[f]) for f in FEATURES], dtype=np.float32)

        # 2) Build mask + mapping from action index -> Action object
        n = int(self.model.action_space.n)
        mask = np.zeros(n, dtype=bool)
        idx_to_action = {}

        for a in playable_actions:
            idx = to_action_space(a)
            if 0 <= idx < n:
                mask[idx] = True
                idx_to_action[idx] = a

        # Safety fallback: if something is off, pick any legal action
        if not idx_to_action:
            return playable_actions[0]

        # 3) Ask PPO for an action (deterministic for “best move” play)
        action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=True)

        return idx_to_action.get(int(action_idx), playable_actions[0])

register_cli_player("PPO", PPOPlayer)
