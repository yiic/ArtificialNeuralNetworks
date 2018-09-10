../obj/proj1/dbg/rand.o: rand.cpp rand.h error.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -g -D_DEBUG -c rand.cpp -o ../obj/proj1/dbg/rand.o
