../obj/proj1/opt/supervised.o: supervised.cpp supervised.h matrix.h error.h string.h \
  rand.h vec.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -O3 -c supervised.cpp -o ../obj/proj1/opt/supervised.o
