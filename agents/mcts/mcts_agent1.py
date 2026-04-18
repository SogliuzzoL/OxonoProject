# MCTS v1 — base
# No value normalization, uniform priors, no tree reuse between turns.
from agent import Agent
from files_99942500_26482510.heuristic import evaluate
from files_99942500_26482510.mcts import MCTS
from files_99942500_26482510.utils import all_possible_actions
from oxono import Game, State


class MCTSAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.mcts = MCTS(
            all_actions=all_possible_actions(),
            eval_fn=evaluate,
            use_normalization=False,
            use_heuristic_priors=False,
        )

    def act(self, state: State, remaining_time):
        for move in Game.actions(state):
            test_state = state.copy()
            Game.apply(test_state, move)
            if Game.is_terminal(test_state):
                return move

        pieces_remaining = state.pieces_x[self.player] + state.pieces_o[self.player]
        time_per_move = remaining_time / max(pieces_remaining, 1)

        self.mcts.run(state, time_simulation=time_per_move, c_puct=1.5, player=self.player)
        return self.mcts.get_best_action()
