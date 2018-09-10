../obj/proj1/dbg/main.o: main.cpp error.h string.h rand.h matrix.h supervised.h baseline.h \
  vec.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -g -D_DEBUG -c main.cpp -o ../obj/proj1/dbg/main.o
