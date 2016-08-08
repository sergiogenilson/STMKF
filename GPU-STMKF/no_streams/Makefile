CPP=g++
CPPFLAGS=$(shell pkg-config --cflags opencv)
LIBS=$(shell pkg-config --libs opencv)

stmkf: main.cpp stmkf.cpp
	$(CPP) -o stmkf main.cpp stmkf.cpp $(CPPFLAGS) $(LIBS) 
