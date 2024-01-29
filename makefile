CXX      = g++
CXXFLAGS = -std=c++17

all: main processing

main: main.cpp eval.cpp search.cpp game.cpp
	${CXX} ${CXXFLAGS} $^ -o $@

processing: processing.cpp
	${CXX} ${CXXFLAGS} $^ -o $@

clean: 
	${RM} main processing a.out *.o *dSYM