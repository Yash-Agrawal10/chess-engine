#pragma once

#include "chess.hpp"
#include "eval.h"

using namespace chess;

int negaMax(Board&, int);

Move bestMove(Board&, int);

int alphaBeta(Board&, int);

Move bestMoveAB(Board&, int);