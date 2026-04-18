# MCTS v5 — v4 + early playout termination with intermediate evaluation.
# Instead of evaluating a leaf node statically, play PLAYOUT_DEPTH random moves
# and evaluate the resulting position. This samples a short trajectory into the
# future, giving a smoother value estimate than pure static evaluation.
from agent import Agent
from files_99942500_26482510.heuristic import evaluate
from files_99942500_26482510.mcts import MCTS
from files_99942500_26482510.utils import all_possible_actions
from oxono import Game, State

TOTEM_X       = "X"
TOTEM_O       = "O"
PLAYOUT_DEPTH = 5


class MCTSAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.mcts = MCTS(
            all_actions=all_possible_actions(),
            eval_fn=evaluate,
            use_normalization=True,
            use_heuristic_priors=True,
            use_early_playout=True,
            playout_depth=PLAYOUT_DEPTH,
        )
        self.last_totem_O = None
        self.last_totem_X = None

    def act(self, state: State, remaining_time):
        for move in Game.actions(state):
            test_state = state.copy()
            Game.apply(test_state, move)
            if Game.is_terminal(test_state):
                return move

        pieces_remaining = state.pieces_x[self.player] + state.pieces_o[self.player]
        time_per_move = remaining_time / max(pieces_remaining, 1)

        if state.last_move is not None:
            if state.totem_O != self.last_totem_O:
                opponent_last_action = (TOTEM_O, state.totem_O, state.last_move)
            else:
                opponent_last_action = (TOTEM_X, state.totem_X, state.last_move)
            self.mcts.update_root(opponent_last_action)

        self.mcts.run(state, time_simulation=time_per_move, c_puct=1.5, player=self.player)

        best_action = self.mcts.get_best_action()
        self.last_totem_O = state.totem_O if best_action[0] != TOTEM_O else best_action[1]
        self.last_totem_X = state.totem_X if best_action[0] != TOTEM_X else best_action[1]
        self.mcts.update_root(best_action)
        return best_action
