CXX      = g++
CXXFLAGS = -std=c++17

all: main

main: main.cpp eval.cpp search.cpp game.cpp
	${CXX} ${CXXFLAGS} $^ -o $@

clean: 
	${RM} main a.out *.o *dSYM