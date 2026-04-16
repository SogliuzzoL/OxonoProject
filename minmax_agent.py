import time

from agent import Agent
from files_99942500_26482510.zobrist import generate_zobrist_table
from oxono import Game

SEARCH_DEPTH = 3
X_SYMBOL = "x"
O_SYMBOL = "o"
MINUS_INFINITY = float("-inf")
PLUS_INFINITY = float("inf")
ZOBRIST_TABLE = generate_zobrist_table()


class SearchTimeout(Exception):
    pass


class MinimaxAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):

        pieces_remaining = state.pieces_x[self.player] + state.pieces_o[self.player]
        move_time_mean = remaining_time / max(pieces_remaining, 1)

        # if few pieces left then we can allow more time per move, max 8sec, else max 4sec
        if pieces_remaining <= 6:
            time_per_move = min(move_time_mean, 8.0)
        else:
            time_per_move = min(move_time_mean, 4.0)

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


##################################################
############## ATTENTION #########################
### COPIÉ COLLÉ DE LA FONCTION DANS MCTS #########
def compute_zobrist_hash(state):
    h = 0
    for row in range(6):
        for col in range(6):
            piece = state.board[row][col]
            if piece is not None:
                symbol, player = piece
                type_idx = (0 if symbol == "o" else 2) + player
                h ^= ZOBRIST_TABLE["pieces"][row][col][type_idx]

    row, col = state.totem_O
    h ^= ZOBRIST_TABLE["totem_O"][row][col]

    row, col = state.totem_X
    h ^= ZOBRIST_TABLE["totem_X"][row][col]

    h ^= ZOBRIST_TABLE["player"][state.current_player]

    return h


##################################################
############## ATTENTION #########################
### COPIÉ COLLÉ DE LA FONCTION DANS MCTS #########


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

    state_hash = compute_zobrist_hash(state)
    if state_hash in transposition_table:
        stocked_score, stocked_depth = transposition_table[state_hash]
        if stocked_depth >= depth:
            return stocked_score

    is_maximizing = state.current_player == player  # True if it's our turn else false
    # Our turn --> choose the move that maximizes our score
    if is_maximizing:
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
            best_score = max(score, best_score)
            # alpha is the best score that I am guaranteed to have
            # we update alpha when there is a greater best_score
            alpha = max(best_score, alpha)
            # if alpha>= beta, min has already better option=this branch will never be chosen --> killer move
            if alpha >= beta:
                killer_moves[depth] = move
                break
        transposition_table[state_hash] = (best_score, depth)
        return best_score
    # Opponent turn --> he chooses the move that minimizes our score
    else:
        worst_score = PLUS_INFINITY
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
                break
        transposition_table[state_hash] = (worst_score, depth)
        return worst_score


def evaluate(state, player):
    """
    Estimate the value of a state for the player
    """
    if Game.is_terminal(state):
        return Game.utility(state, player) * 10000
    player_score = 0
    player_score += score_all_lines(state, player)
    player_score += score_immediate_threats(state, player)

    return player_score


def score_all_lines(state, player):
    """
    Score for rows and columns checking color/symbol alignments
    """
    board = state.board
    total = 0

    for row in board:
        total += score_line(row, player)

    for col in range(len(board[0])):
        column = [board[row][col] for row in range(len(board))]
        total += score_line(column, player)

    return total


def score_line(line, player):
    """
    Create a score of a line (row or col) for a player
    """
    score = 0

    number_consecutive_player_pieces = 0
    number_consecutive_opponent_pieces = 0
    number_consecutive_x_pieces = 0
    number_consecutive_y_pieces = 0

    for cell in line:
        # Update based on color
        if cell is not None and cell[1] == player:
            number_consecutive_player_pieces += 1
            number_consecutive_opponent_pieces = 0
        elif cell is not None and cell[1] != player:
            number_consecutive_opponent_pieces += 1
            number_consecutive_player_pieces = 0
        else:
            number_consecutive_player_pieces = 0
            number_consecutive_opponent_pieces = 0

        # Update based on symbol
        if cell is not None and cell[0] == X_SYMBOL:
            number_consecutive_x_pieces += 1
            number_consecutive_y_pieces = 0
        elif cell is not None and cell[0] == O_SYMBOL:
            number_consecutive_y_pieces += 1
            number_consecutive_x_pieces = 0
        else:
            number_consecutive_x_pieces = 0
            number_consecutive_y_pieces = 0

        # Score for consecutive color pieces
        if number_consecutive_player_pieces == 2:
            score += 10
        elif number_consecutive_player_pieces == 3:
            score += 100
        if number_consecutive_opponent_pieces == 2:
            score -= 10
        elif number_consecutive_opponent_pieces == 3:
            score -= 120

        # Score for consecutive symbol pieces
        if number_consecutive_x_pieces == 2:
            score += 5
        elif number_consecutive_x_pieces == 3:
            score += 30
        if number_consecutive_y_pieces == 2:
            score += 5
        elif number_consecutive_y_pieces == 3:
            score += 30

    return score


def score_immediate_threats(state, player):
    """
    Detect windows that are one move away from a winning pattern
    """
    board = state.board
    opponent = get_opponent(player)
    score = 0
    player_threat_count = 0
    opponent_threat_count = 0

    for row in board:
        line_score, player_threats, opponent_threats = score_threats_in_line(
            row, player, opponent
        )
        score += line_score
        player_threat_count += player_threats
        opponent_threat_count += opponent_threats

    for col in range(len(board[0])):
        column = [board[row][col] for row in range(len(board))]
        line_score, player_threats, opponent_threats = score_threats_in_line(
            column, player, opponent
        )
        score += line_score
        player_threat_count += player_threats
        opponent_threat_count += opponent_threats

    # if one player has multiple threats possibilites
    if player_threat_count >= 2:
        score += 5000
    if opponent_threat_count >= 2:
        score -= 6000

    return score


def score_threats_in_line(line, player, opponent):

    score = 0
    player_threat_count = 0
    opponent_threat_count = 0
    # on check fenetre de 4 cases donc len-3
    # soit 0 à 3, soit 1 À 4, soit 2 À 5
    for start in range(len(line) - 3):
        window = line[start : start + 4]

        empty_cell_count = sum(1 for cell in window if cell is None)
        # on check uniquement les fenetres de cases avec exactement 1 case vide
        if empty_cell_count != 1:
            continue

        non_empty_cells_window = [cell for cell in window if cell is not None]

        # color threats
        player_pieces_window_count = sum(
            1 for cell in non_empty_cells_window if cell[1] == player
        )
        opponent_pieces_window_count = sum(
            1 for cell in non_empty_cells_window if cell[1] == opponent
        )

        if player_pieces_window_count == 3:
            score += 800
            player_threat_count += 1
        if opponent_pieces_window_count == 3:
            score -= 900
            opponent_threat_count += 1

        # symbol threats
        x_pieces_window_count = sum(
            1 for cell in non_empty_cells_window if cell[0] == X_SYMBOL
        )
        o_pieces_window_count = sum(
            1 for cell in non_empty_cells_window if cell[0] == O_SYMBOL
        )

        if x_pieces_window_count == 3:
            score += 120
            player_threat_count += 1
        if o_pieces_window_count == 3:
            score += 120
            player_threat_count += 1

    return score, player_threat_count, opponent_threat_count


def get_opponent(player):
    return 1 - player
