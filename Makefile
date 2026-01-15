# MPILS: Multi-Phase Iterated Local Search

# Author: Youri Rigaud
# License: GNU GPLv3

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -fno-omit-frame-pointer

INCLUDE_DIRS = -Iinclude
SRC_DIR = src
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/objects

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BUILD_DIR)/mpils

CPLEX_DIR = /opt/ibm/ILOG/CPLEX_Studio2212

INCLUDE_DIRS = \
    -Iinclude \
    -I$(CPLEX_DIR)/cplex/include \
    -I$(CPLEX_DIR)/concert/include

LIB_DIRS = \
    -L$(CPLEX_DIR)/cplex/lib/x86-64_linux/static_pic \
    -L$(CPLEX_DIR)/concert/lib/x86-64_linux/static_pic

LIBS = -lilocplex -lcplex -lconcert -lm -lpthread -larmadillo



.PHONY: all clean
all: $(TARGET)
$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_DIRS) $(LIBS)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@
$(BUILD_DIR):
	mkdir -p $@
$(OBJ_DIR):
	mkdir -p $@
clean:
	rm -rf $(BUILD_DIR)