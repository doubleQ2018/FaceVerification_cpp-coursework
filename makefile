CXX = g++ 
CXXFLAGS = -std=c++11
CXXFLAGS += `pkg-config --cflags --libs opencv`
CXX_SRCS := $(shell find src/ -name "*.cpp" ! -name "lbp.cpp")
start:
	$(CXX) $(CXXFLAGS) -o run_adaboost main.cpp $(CXX_SRCS)
