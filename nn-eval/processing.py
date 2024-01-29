import numpy as np

piecemap = {
    'K': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'P': 5,
    'p': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'k': 11
}

def fen_to_worldviews(fen: str):
    # convert FEN string to my / their worldviews

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

    # flip worldview based on to_move
    if white_to_move:
        their_worldview = np.flip(their_worldview)
    else:
        my_worldview = np.flip(my_worldview)

    # return
    return [my_worldview, their_worldview]

def print_worldview(worldview):
    # print out worldview
    for key in piecemap:
        index = piecemap[key]
        pieceboard = worldview[index]
        pieceboard = np.reshape(pieceboard, (8, 8))
        print(key, index)
        print(pieceboard)
        print("\n")

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
    
    my_product = np.outer(my_king, my_others).flatten()
    print(np.shape(my_king), np.shape(my_others), np.shape(my_product))
    their_product = np.outer(their_king, their_others).flatten()

    halfkp = np.concatenate([my_product, their_product])
    return halfkp

def fen_to_halfkp(fen: str):
    my_worldview, their_worldview = fen_to_worldviews(fen)
    halfkp = worldviews_to_halfkp(my_worldview, their_worldview)
    return halfkp

fen = input("Enter FEN: ")
halfkp = fen_to_halfkp(fen)
print(np.shape(halfkp))