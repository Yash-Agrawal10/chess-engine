#include <iostream>
#include <map>

using namespace std;

array<array<int, 64>, 12> fen_to_piece_board(char *fen)
{
    // piece map of fen string to array index int
    map<char, int> piece_map = {
        {'P', 0},
        {'N', 1},
        {'B', 2},
        {'R', 3},
        {'Q', 4},
        {'K', 5},
        {'k', 6},
        {'q', 7},
        {'r', 8},
        {'b', 9},
        {'n', 10},
        {'p', 11}};

    // initialize unflattened pieceboards
    // 12 x 8 x 8 (for 12 pieces, 8 x 8 chess board)
    array<array<int, 64>, 12> output = {};

    // interpret fen string
    int curr_row = 0;
    int curr_col = 0;
    int curr_square = 0;
    bool white_to_move;
    for (int i = 0; i < 1000; i++) // 1000 is arbitary number
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
            output[piece][curr_square] = 1;
            curr_col += 1;
        }
    }

    // reverse if black to move
    if (!white_to_move)
    {

        for (int i = 0; i < 12; i++)
        {
            reverse(output[i].begin(), output[i].end());
        }
        reverse(output.begin(), output.end());
    }

    return output;
}

int main()
{
    char *fen;
    printf("Enter FEN code: ");
    scanf("%[^\n]", fen);
    cout << "\n"
         << fen;
    printf("\n");

    array<array<int, 64>, 12> board_white = fen_to_piece_board(fen);

    // order is my p, n, b, r, q, k, then their reverse
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                cout << board_white[i][j * 8 + k] << " ";
            }
            cout << "\n";
        }
        cout << "\n\n";
    }
}