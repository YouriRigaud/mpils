# MPILS: Multi-Phase Iterated Local Search

# Author: Youri Rigaud
# License: GNU GPLv3

CXX_SEQ = g++
CXX_MPI = /usr/lib64/openmpi/bin/mpic++

CXX = $(CXX_SEQ)
CXXFLAGS = -std=c++17 -Wall -Wextra -g -fno-omit-frame-pointer

MPI_FLAGS = -DUSE_MPI

INCLUDE_DIRS = -Iinclude
SRC_DIR = src
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/objects

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BUILD_DIR)/mpils

CPLEX_DIR = /home/ibm/cplex-studio/22.1.2

INCLUDE_DIRS = \
    -Iinclude \
    -I$(CPLEX_DIR)/cplex/include \
    -I$(CPLEX_DIR)/concert/include

LIB_DIRS = \
    -L$(CPLEX_DIR)/cplex/lib/x86-64_linux/static_pic \
    -L$(CPLEX_DIR)/concert/lib/x86-64_linux/static_pic

LIBS = -lilocplex -lcplex -lconcert -lm -lpthread -larmadillo \
       -lrt -lbz2 -ldl -lstdc++ -lopenblas -lgfortran -lz



.PHONY: all clean mpi
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
mpi: CXX = $(CXX_MPI)
mpi: CXXFLAGS += $(MPI_FLAGS)
mpi: all