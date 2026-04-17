import time

from agent import Agent
from files_99942500_26482510.zobrist import compute_zobrist_hash, generate_zobrist_table
from files_99942500_26482510.heuristic import evaluate
from oxono import Game


MINUS_INFINITY = float("-inf")
PLUS_INFINITY = float("inf")
ZOBRIST_TABLE = generate_zobrist_table()


class SearchTimeout(Exception):
    pass


class MinimaxAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        t0 = time.time()
        for move in Game.actions(state):
            test_state = state.copy()
            Game.apply(test_state, move)
            if Game.is_terminal(test_state):
                return move

        t1 = time.time()
        elapsed_time = t1 - t0
        remaining_time -= elapsed_time

        pieces_remaining = state.pieces_x[self.player] + state.pieces_o[self.player]
        move_time_mean = remaining_time / max(pieces_remaining, 1)

        # if few pieces left then we can allow more time per move, max 8sec, else max 4sec
        # if pieces_remaining <= 6:
        #     time_per_move = min(move_time_mean, 8.0)
        # else:
        #     time_per_move = min(move_time_mean, 4.0)
        time_per_move = move_time_mean

        depth = 1

        ## on met un best_move de sécurité dans le cas ou dans le temps imparti il trv pas de move, au moins
        # on fait un move comme ca il perd pas
        best_move = list(Game.actions(state))[0]

        deadline = time.time() + max(0.0, time_per_move)

        killer_moves = {}
        transposition_table = {}

        while True:
            if (time.time()) >= deadline:
                break
            try:
                current_best_move = search_best_move(
                    state,
                    depth,
                    self.player,
                    deadline,
                    killer_moves,
                    transposition_table,
                )
                # màj seulement si toute la profondeur est terminée
                if current_best_move is not None:
                    best_move = current_best_move

                depth += 1
            except SearchTimeout:
                break
        return best_move


def search_best_move(state, depth, player, deadline, killer_moves, transposition_table):
    actions = list(Game.actions(state))
    if not actions:
        return None

    alpha = MINUS_INFINITY
    beta = PLUS_INFINITY

    best_move = None
    best_score = MINUS_INFINITY

    for move in get_ordered_moves(state, player, depth, killer_moves):
        check_timeout(deadline)

        new_state = state.copy()
        Game.apply(new_state, move)

        score = minmax_with_apha_beta_pruning(
            new_state,
            depth - 1,
            player,
            alpha,
            beta,
            deadline,
            killer_moves,
            transposition_table,
        )

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)

    return best_move


def get_ordered_moves(state, player, depth, killer_moves):
    moves_ordered_by_score = []
    moves_by_score = []  # liste sous forme (score, move)
    killer_move = killer_moves.get(depth)
    all_moves = list(Game.actions(state))
    # si il existe un killer move pr cette profondeur, alors on le met d'abord car fait bon score
    if killer_move != None and killer_move in all_moves:
        moves_ordered_by_score.append(killer_move)
        all_moves.remove(killer_move)

    for move in all_moves:
        new_state = state.copy()
        Game.apply(new_state, move)
        score = evaluate(new_state, player)

        if Game.is_terminal(new_state):
            score += 100000

        moves_by_score.append((score, move))

    # sort pour mettre meilleurs coups devant
    moves_by_score.sort(key=lambda x: x[0], reverse=True)

    # les moves triés sont placés derrière killer move
    for score, move in moves_by_score:
        moves_ordered_by_score.append(move)

    return moves_ordered_by_score


def check_timeout(deadline):
    if time.time() >= deadline:
        raise SearchTimeout()


def minmax_with_apha_beta_pruning(
    state, depth, player, alpha, beta, deadline, killer_moves, transposition_table
):
    """
    Recursively evaluate a game state using the MinMax algo
    Returns the best possible score for the player assuming both players play optimally
    """
    check_timeout(deadline)

    # game finished
    if Game.is_terminal(state):
        return Game.utility(state, player) * 10000
    if depth == 0:
        return evaluate(state, player)

    state_hash = compute_zobrist_hash(ZOBRIST_TABLE, state)
    if state_hash in transposition_table:
        stocked_score, stocked_depth, flag_bound = transposition_table[state_hash]
        if stocked_depth >= depth:
            if flag_bound == "exact":
                return stocked_score
            elif flag_bound == "lower":
                alpha = max(alpha, stocked_score)
            elif flag_bound == "upper":
                beta = min(beta, stocked_score)
            if alpha >= beta:
                return stocked_score

    is_maximizing = state.current_player == player  # True if it's our turn else false
    # Our turn --> choose the move that maximizes our score
    if is_maximizing:
        best_score = MINUS_INFINITY
        cutoff= False
        for move in get_ordered_moves(state, player, depth, killer_moves):
            check_timeout(deadline)
            new_state = state.copy()
            Game.apply(new_state, move)
            score = minmax_with_apha_beta_pruning(
                new_state,
                depth - 1,
                player,
                alpha,
                beta,
                deadline,
                killer_moves,
                transposition_table,
            )
            best_score = max(score, best_score)
            # alpha is the best score that I am guaranteed to have
            # we update alpha when there is a greater best_score
            alpha = max(best_score, alpha)
            # if alpha>= beta, min has already better option=this branch will never be chosen --> killer move
            if alpha >= beta:
                killer_moves[depth] = move
                cutoff=True
                break
        if cutoff:
            flag_bound="lower"
        else:
            flag_bound="exact"
        transposition_table[state_hash] = (best_score, depth,flag_bound)
        return best_score
    # Opponent turn --> he chooses the move that minimizes our score
    else:
        worst_score = PLUS_INFINITY
        cutoff = False
        for move in get_ordered_moves(state, player, depth, killer_moves):
            check_timeout(deadline)

            new_state = state.copy()
            Game.apply(new_state, move)
            score = minmax_with_apha_beta_pruning(
                new_state,
                depth - 1,
                player,
                alpha,
                beta,
                deadline,
                killer_moves,
                transposition_table,
            )
            worst_score = min(score, worst_score)
            # beta is the best score that my opponenet is guaranteed to have
            # we update beta when there is a smaller worst_score
            beta = min(beta, worst_score)
            # if beta <= alpha, max has already better option=this branch will never be chosen --> killer move
            if beta <= alpha:
                killer_moves[depth] = move
                cutoff = True
                break
        if cutoff:
            flag_bound="upper"
        else:
            flag_bound="exact"
        transposition_table[state_hash] = (worst_score, depth,flag_bound)
        return worst_score


