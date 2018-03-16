../obj/proj1/dbg/error.o: error.cpp error.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -g -D_DEBUG -c error.cpp -o ../obj/proj1/dbg/error.o
