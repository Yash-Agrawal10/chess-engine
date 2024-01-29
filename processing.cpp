#include "processing.h"

// Constants
// Capital is mine, lower is theirs (based on to move)
map<char, int> piece_map = {
    {'K', 0},
    {'N', 1},
    {'B', 2},
    {'R', 3},
    {'Q', 4},
    {'P', 5},
    {'p', 6},
    {'q', 7},
    {'r', 8},
    {'b', 9},
    {'n', 10},
    {'k', 11}};

// Helpers
array<array<int, 64>, 12> reverse_worldview(array<array<int, 64>, 12> worldview)
{
    for (int i = 0; i < 12; i++)
    {
        reverse(worldview[i].begin(), worldview[i].end());
    }
    reverse(worldview.begin(), worldview.end());
    return worldview;
}

// Computations
array<array<array<int, 64>, 12>, 2> fen_to_piece_boards(char *fen)
{
    // initialize unflattened pieceboards
    // 12 x 8 x 8 (for 12 pieces, 8 x 8 chess board)
    array<array<int, 64>, 12> my_worldview = {};
    array<array<int, 64>, 12> their_worldview = {};

    // interpret fen string
    int curr_row = 0;
    int curr_col = 0;
    int curr_square = 0;
    bool white_to_move;
    // 75 is slightly greater than max relevant fen length
    for (int i = 0; i < 75; i++)
    {
        if (fen[i] == '/')
        {
            curr_row += 1;
            curr_col = 0;
        }
        else if (isdigit(fen[i]))
        {
            int c = fen[i] - '0';
            curr_col += c;
        }
        else if (fen[i] == ' ')
        {
            i++;
            if (fen[i] == 'w')
            {
                white_to_move = true;
            }
            else
            {
                white_to_move = false;
            }
            break;
        }
        else
        {
            int piece = piece_map[fen[i]];
            curr_square = curr_row * 8 + curr_col;
            my_worldview[piece][curr_square] = 1;
            their_worldview[piece][curr_square] = 1;
            curr_col += 1;
        }
    }

    // reverse one worldview
    if (white_to_move)
    {
        their_worldview = reverse_worldview(their_worldview);
    }
    else
    {
        my_worldview = reverse_worldview(my_worldview);
    }

    // combine two worldviews and return
    array<array<array<int, 64>, 12>, 2> output;
    output[0] = my_worldview;
    output[1] = their_worldview;
    return output;
}

int fen_to_halfkp(char *fen)
{
    array<array<array<int, 64>, 12>, 2> piece_boards = fen_to_piece_boards(fen);
    // create my-king, my-worldview, their-king, and their-worldview arrays
    // do matrix multiplication (king x worldview)
    // flatten outputs
    // concatenate outputs (mine/to-move first)
    // return this as output, halfkp!!
    return 0;
}

// Print Functions
void print_piece_boards(array<array<int, 64>, 12> piece_boards)
{
    map<char, int>::iterator it = piece_map.begin();
    while (it != piece_map.end())
    {
        char piece_type = it->first;
        int index = it->second;
        printf("Piece %c, Index %i\n", piece_type, index);
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                cout << piece_boards[index][j * 8 + k] << " ";
            }
            cout << "\n";
        }
        cout << "\n\n";
        it++;
    }
}