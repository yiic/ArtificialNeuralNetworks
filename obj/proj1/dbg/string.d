../obj/proj1/dbg/string.o: string.cpp string.h error.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -g -D_DEBUG -c string.cpp -o ../obj/proj1/dbg/string.o
