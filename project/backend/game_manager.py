# game_manager.py

from catanatron import Game
from PLAYMYMODEL import PPOPlayer


class GameManager:
    def __init__(self):
        self.game = None
        self.human_color = "red"
        self.bot_color = "blue"

    def start_game(self, model_path: str):
        bot = PPOPlayer(self.bot_color, model_path=model_path)

        # We include a dummy player slot for human
        # Human moves will be injected manually
        self.game = Game(players=[bot])

        return {"status": "Game started"}

    def get_state(self):
        if not self.game:
            return {"error": "No game running"}

        return self.game.serialize()

    def get_playable_actions(self):
        if not self.game:
            return {"error": "No game running"}

        actions = self.game.state.playable_actions
        return {
            "actions": [str(a) for a in actions]
        }

    def apply_human_move(self, action_index: int):
        if not self.game:
            return {"error": "No game running"}

        playable = self.game.state.playable_actions

        if action_index < 0 or action_index >= len(playable):
            return {"error": "Invalid move index"}

        action = playable[action_index]
        self.game.play(action)

        return {"status": "Move applied"}
