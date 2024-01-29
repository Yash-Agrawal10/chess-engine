#include <iostream>
#include <map>

using namespace std;

extern map<char, int> piece_map;

array<array<int, 64>, 12> reverse_worldview(array<array<int, 64>, 12>);

array<array<array<int, 64>, 12>, 2> fen_to_piece_boards(char *);

int fen_to_halfkp(char *);

void print_piece_boards(array<array<int, 64>, 12>);