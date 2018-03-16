../obj/proj1/opt/matrix.o: matrix.cpp matrix.h rand.h error.h string.h vec.h json.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -O3 -c matrix.cpp -o ../obj/proj1/opt/matrix.o
