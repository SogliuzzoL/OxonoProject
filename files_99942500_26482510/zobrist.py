import random


def generate_zobrist_table():
    random.seed(42)
    table = {
        "pieces": [
            [[random.getrandbits(64) for _ in range(4)] for _ in range(6)]
            for _ in range(6)
        ],
        "totem_O": [[random.getrandbits(64) for _ in range(6)] for _ in range(6)],
        "totem_X": [[random.getrandbits(64) for _ in range(6)] for _ in range(6)],
        "player": [random.getrandbits(64) for _ in range(2)],
    }
    return table
