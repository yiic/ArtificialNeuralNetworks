../obj/proj1/dbg/json.o: json.cpp json.h error.h string.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -g -D_DEBUG -c json.cpp -o ../obj/proj1/dbg/json.o
