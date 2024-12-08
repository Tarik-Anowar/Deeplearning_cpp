# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -g -fopenmp

# Library files (source files from lib folder)
LIB_SRC = \
    lib/model.cpp \
	lib/layer.cpp \
    lib/computation_graph.cpp \
    lib/optimizer.cpp \
    lib/activation.cpp \
	lib/node.cpp \
    lib/loss.cpp

# Object files for library sources
LIB_OBJ = $(LIB_SRC:.cpp=.o)

# Test source file
TEST_SRC = test.cpp

# Object file for the test
TEST_OBJ = test.o

# Output executable
TARGET = test_program

# Rule to compile .cpp files in lib/ folder (only when they change)
lib/%.o: lib/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile the test.cpp file
$(TEST_OBJ): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -c $(TEST_SRC) -o $(TEST_OBJ)

# Rule to create the executable
$(TARGET): $(LIB_OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $(LIB_OBJ) $(TEST_OBJ) -o $(TARGET)

# Clean the build
clean:
	rm -f $(LIB_OBJ) $(TEST_OBJ) $(TARGET)

# Compile only lib files (run `make lib`)
lib: $(LIB_OBJ)

# Compile only test file (run `make test`)
test: $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $(TEST_OBJ) $(LIB_OBJ) -o $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)
