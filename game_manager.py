# game_manager.py

from catanatron import Game
from catanatron.gui import run_gui
from catanatron.players.weighted_random import WeightedRandomPlayer

from PLAYMYMODEL import PPOPlayer


class GameManager:
    def __init__(self):
        self.game = None
        self.human_color = None
        self.bot_color = None
        self.gui_running = False

    # ---------------------------------------------------
    # START GAME
    # ---------------------------------------------------
    def start_game(self, model_path: str):
        """
        Starts a new game with:
        - Human player (handled via API)
        - PPO bot player
        """

        self.human_color = "red"
        self.bot_color = "blue"

        # Create bot player
        bot = PPOPlayer(self.bot_color, model_path=model_path)

        # Temporary placeholder for human
        # We will manually inject human moves via API
        human = WeightedRandomPlayer(self.human_color)

        # Create game
        self.game = Game(players=[human, bot])

        return {"status": "Game started"}

    # ---------------------------------------------------
    # GET CURRENT GAME STATE
    # ---------------------------------------------------
    def get_state(self):
        if not self.game:
            return {"error": "No game running"}

        return self.game.serialize()

    # ---------------------------------------------------
    # APPLY HUMAN MOVE
    # ---------------------------------------------------
    def apply_human_move(self, action_index: int):
        if not self.game:
            return {"error": "No game running"}

        playable = self.game.state.playable_actions

        if not playable:
            return {"error": "No playable actions"}

        if action_index < 0 or action_index >= len(playable):
            return {"error": "Invalid move index"}

        action = playable[action_index]

        self.game.play(action)

        return {"status": "Move applied"}

    # ---------------------------------------------------
    # LAUNCH GUI
    # ---------------------------------------------------
    def launch_gui(self):
        """
        Launches Catanatron GUI in a separate window.
        Safe to call once after starting the game.
        """
        if not self.game:
            return {"error": "No game running"}

        if self.gui_running:
            return {"status": "GUI already running"}

        self.gui_running = True
        run_gui(self.game)

        return {"status": "GUI launched"}
