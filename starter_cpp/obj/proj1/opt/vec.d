../obj/proj1/opt/vec.o: vec.cpp vec.h error.h rand.h matrix.h string.h json.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -O3 -c vec.cpp -o ../obj/proj1/opt/vec.o
