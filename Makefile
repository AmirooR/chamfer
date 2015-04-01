OPENCV_CFLAGS=`pkg-config --cflags opencv`
OPENCV_LFLAGS=`pkg-config --libs opencv`

all: chamfer
chamfer: chamfer.cpp TemplateMatcher.o precomp.o chamfermatching.o 
	g++  precomp.o chamfermatching.o TemplateMatcher.o chamfer.cpp  ${OPENCV_CFLAGS} ${OPENCV_LFLAGS} -o chamfer
TemplateMatcher.o: TemplateMatcher.cpp TemplateMatcher.hpp
	g++ -c ${OPENCV_CFLAGS} TemplateMatcher.cpp -o TemplateMatcher.o
precomp.o: precomp.hpp precomp.cpp
	g++ -c ${OPENCV_CFLAGS} precomp.cpp -o precomp.o
chamfermatching.o: chamfermatching.cpp
	g++ -c ${OPENCV_CFLAGS} chamfermatching.cpp -o chamfermatching.o
clean:
	rm *.o
