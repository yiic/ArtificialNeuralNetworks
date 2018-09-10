../obj/proj1/opt/baseline.o: baseline.cpp baseline.h matrix.h vec.h error.h rand.h \
  supervised.h
	g++ -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -std=c++11 -O3 -c baseline.cpp -o ../obj/proj1/opt/baseline.o
