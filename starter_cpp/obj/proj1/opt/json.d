../obj/proj1/opt/json.o: json.cpp json.h error.h string.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -O3 -c json.cpp -o ../obj/proj1/opt/json.o
