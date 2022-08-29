
default:	discBar.o

%.o:		%.cc %.h
		$(CXX) -c $< -o $@ -Wall -std=c++17

clean:
		-rm -f discBar.o
