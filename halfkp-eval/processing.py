import numpy as np

piecemap = {
    'K': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'P': 5,
    'p': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'k': 11
}

def fen_to_worldviews(fen: str):
    # convert FEN string to [my_worldview, their_worldview, white_to_move]
    # piece order in worldviews based on global piecemap

    # initialize variables
    my_worldview = np.zeros((12, 64))
    their_worldview = np.zeros((12, 64))
    curr_row = 0
    curr_col = 0
    curr_square = 0
    white_to_move = False

    # interpret FEN string
    for i, char in enumerate(fen):
        if char == '/':
            curr_row += 1
            curr_col = 0
        elif char.isdigit():
            curr_col += int(char)
        elif char == ' ':
            i += 1
            if fen[i] == 'w':
                white_to_move = True
            break
        else:
            piece_index = piecemap[char]
            curr_square = curr_row * 8 + curr_col
            my_worldview[piece_index][curr_square] = 1
            their_worldview[piece_index][curr_square] = 1
            curr_col += 1

    # flip worldview based on to_move (currently ignored, white is my_worldview)
    if white_to_move:
        their_worldview = np.flip(their_worldview)
    else:
        my_worldview = np.flip(my_worldview)

    # return
    return [my_worldview, their_worldview, white_to_move]

def worldviews_to_halfkp(my_worldview, their_worldview):
    # converts worldviews (where own king is index 0, their king is index 11)
    # into halfkp structure

    my_king = my_worldview[0]
    their_king = their_worldview[0]
    my_others = []
    their_others = []
    for i in range(1, 11):
        my_others.append(my_worldview[i])
        their_others.append(their_worldview[i])
    my_others = np.concatenate(my_others)
    their_others = np.concatenate(their_others)
    
    my_product = np.outer(my_king, my_others).flatten().astype(np.single)
    their_product = np.outer(their_king, their_others).flatten().astype(np.single)

    return [my_product, their_product]

def normalize_eval(eval: int, white_to_move: bool):
    # flip eval if black to move
    # and bound eval to [-15, 15]
    if not(white_to_move):
        eval = eval * -1
    if eval < -15:
        bounded_eval = -15
    elif eval > 15:
        bounded_eval = 15
    else:
        bounded_eval = eval
    # return bounded_eval, not eval
    return eval

def denormalize_eval(eval: int, white_to_move: bool):
    # flip eval if black to move
    if not(white_to_move):
        eval = eval * -1
    return eval

def process_row(row):
    # process a row of the dataframe
    fen = row["fen"]
    stockfish_eval = row["eval"]
    my_worldview, their_worldview, white_to_move = fen_to_worldviews(fen)
    mine, theirs = worldviews_to_halfkp(my_worldview, their_worldview)
    row["mine"] = mine
    row["theirs"] = theirs
    row["white_to_move"] = white_to_move
    normalized_eval = normalize_eval(stockfish_eval, white_to_move)
    row["normalized_eval"] = np.array([normalized_eval], dtype=np.single)
    return row