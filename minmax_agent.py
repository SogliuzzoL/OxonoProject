from agent import Agent
from oxono import Game

SEARCH_DEPTH = 3
X_SYMBOL = 'x'
O_SYMBOL= 'o'
MINUS_INFINITY = float('-inf') 
PLUS_INFINITY = float('inf') 
class MinimaxAgent(Agent):
    

    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        best_score = MINUS_INFINITY 
        best_move = None
        alpha = MINUS_INFINITY #alpha is best score for maximum
        beta= PLUS_INFINITY # beta is best score for minimum
        for move in Game.actions(state):
                new_state = state.copy()
                Game.apply(new_state, move)
                score = minmax_with_apha_beta_pruning(new_state, SEARCH_DEPTH, self.player, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move 



       

def minmax_with_apha_beta_pruning(state, depth, player, alpha, beta):
    """
    Recursively evaluate a game state using the MinMax algo
    Returns the best possible score for the player assuming both players play optimally
    """
    is_maximizing = (state.current_player==player) # True if it's our turn else false
    # game finished
    if Game.is_terminal(state):
        return Game.utility(state, player)
    if depth==0 : 
        return evaluate(state, player)
    # Our turn --> choose the move that maximizes our score
    if is_maximizing: 
        best_score = MINUS_INFINITY
        for move in Game.actions(state):
            new_state = state.copy()
            Game.apply(new_state, move)
            score=minmax_with_apha_beta_pruning(new_state,depth-1,player,alpha,beta)
            best_score = max(score, best_score)
            #alpha is the best score that I am guaranteed to have
            #we update alpha when there is a greater best_score
            alpha = max(best_score,alpha)
            #if my best score is greater than what opponent tolerates, he will never comme in this branch
            if best_score >= beta :
                return best_score # cut this branch
        return best_score
    #Opponent turn --> he chooses the move that minimizes our score
    else : 
        worst_score = PLUS_INFINITY
        for move in Game.actions(state):
            new_state = state.copy()
            Game.apply(new_state, move)
            score=minmax_with_apha_beta_pruning(new_state,depth-1,player,alpha,beta)
            worst_score = min(score,worst_score)
            #beta is the best score that my opponenet is guaranteed to have
            #we update beta when there is a smaller worst_score
            beta= min(beta, worst_score)
            #if my worst score is smaller than what opponent tolerates, he will never comme in this branch
            if worst_score <= alpha:
                return worst_score # cut this branch
        return worst_score


def count_sequence(line, player):
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
            score -= 100

        # Score for consecutive symbol pieces
        if number_consecutive_x_pieces == 2:
            score += 10
        elif number_consecutive_x_pieces == 3:
            score += 50
        if number_consecutive_y_pieces == 2:
            score += 10
        elif number_consecutive_y_pieces == 3:
            score += 50

    return score


def evaluate(state, player):
    """
    Estimate the value of a state for the player
    """
    player_score = 0

    # Calculate score for all rows and columns
    for index in range(6):
        row = [state.board[index][col] for col in range(6)]
        player_score += count_sequence(row, player)

        column = [state.board[row_index][index] for row_index in range(6)]
        player_score += count_sequence(column, player)

    return player_score
