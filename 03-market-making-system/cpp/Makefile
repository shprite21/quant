CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
TARGET ?= benchmark
SOURCES = benchmark.cpp black_scholes.cpp greeks.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) 200000 123

clean:
	rm -f $(TARGET) $(TARGET).exe *.o
