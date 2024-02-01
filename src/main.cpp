#include "chess.hpp"
#include "eval.h"
#include "search.h"
#include "game.h"
#include "processing.h"

using namespace chess;

int main()
{

    std::array<int, 5> arr1 = {1, 2, 3, 4, 5};
    std::array<int, 5> arr2 = {6, 7, 8, 9, 10};
    std::array<int, 5> arr3 = {11, 12, 13, 14, 15};

    /*
    // get input FEN
    char fen[100];
    printf("Enter FEN code: ");
    scanf("%[^\n]s", fen);
    printf("\n");

    // compute and print piece boards for FEN
    array<array<array<int, 64>, 12>, 2> output = fen_to_piece_boards(fen);
    printf("My Worldview: \n");
    print_piece_boards(output[0]);
    printf("\n\n\nTheir Worldview \n");
    print_piece_boards(output[1]);
    return 0;
    */
}