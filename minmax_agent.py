from agent import Agent
from oxono import Game

SEARCH_DEPTH = 3
X_SYMBOL = 'x'
O_SYMBOL= 'o'
class MinimaxAgent(Agent):
    

    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        best_score = float('-inf')
        best_move = None
        for move in Game.actions(state):
                new_state = state.copy()
                Game.apply(new_state, move)
                score = minmax(new_state, SEARCH_DEPTH, self.player)
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move 



       

def minmax(state, depth, player):
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
        best_score = float('-inf') # -infinity
        #is_maximizing=False
        for move in Game.actions(state):
            new_state = state.copy()
            Game.apply(new_state, move)
            score=minmax(new_state,depth-1,player)
            best_score = max(score, best_score)
        return best_score
    #Opponent turn --> he chooses the move that minimizes our score
    else : 
        worst_score = float('inf') # + infinity
        #is_maximizing=True
        for move in Game.actions(state):
            new_state = state.copy()
            Game.apply(new_state, move)
            score=minmax(new_state,depth-1,player)
            worst_score = min(score,worst_score)
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
