../obj/proj1/opt/rand.o: rand.cpp rand.h error.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -O3 -c rand.cpp -o ../obj/proj1/opt/rand.o
