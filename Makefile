# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -g

# Source files
SRC = test.cpp \
      a/model.cpp \
      lib/backpropagation.cpp \
      lib/optimizer.cpp \
      lib/activation.cpp \
      lib/layer.cpp \
      lib/loss.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Output executable
TARGET = test_program

# Default target
all: $(TARGET)

# Rule to create the executable
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET)

# Rule to compile .cpp files to .o object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean the build
clean:
	rm -f $(OBJ) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)
