../obj/proj1/opt/main.o: main.cpp error.h string.h rand.h matrix.h supervised.h baseline.h \
  vec.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -O3 -c main.cpp -o ../obj/proj1/opt/main.o
