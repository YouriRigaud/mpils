# MPILS: Multi-Phase Iterated Local Search

# Author: Youri Rigaud
# License: GNU GPLv3

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

INCLUDE_DIRS = -Iinclude
SRC_DIR = src
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/objects

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BUILD_DIR)/mpils

.PHONY: all clean
all: $(TARGET)
$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@
$(BUILD_DIR):
	mkdir -p $@
$(OBJ_DIR):
	mkdir -p $@
clean:
	rm -rf $(BUILD_DIR)