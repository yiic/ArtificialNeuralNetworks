../obj/proj1/opt/string.o: string.cpp string.h error.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -O3 -c string.cpp -o ../obj/proj1/opt/string.o
