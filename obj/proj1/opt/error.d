../obj/proj1/opt/error.o: error.cpp error.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -O3 -c error.cpp -o ../obj/proj1/opt/error.o
