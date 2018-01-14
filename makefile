CXX = g++ -std=c++11
CXXFLAGS = `pkg-config --cflags --libs opencv`
start:
	$(CXX) $(CXXFLAGS) -o main.o main.cpp adaboost.cpp get_feature.cpp
