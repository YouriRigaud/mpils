# MPILS: Multi-Phase Iterated Local Search

# Author: Youri Rigaud
# License: GNU GPLv3

CXX_SEQ = g++
CXX_MPI = mpic++

CXX = $(CXX_SEQ)
CXXFLAGS = -std=c++17 -Wall -Wextra -g -fno-omit-frame-pointer

MPI_FLAGS = -DUSE_MPI

SRC_DIR = src
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/objects

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BUILD_DIR)/mpils

#CPLEX_DIR = /home/yorig/CPLEX_Studio2212

# MLPack
mlpack_dir := /home/yorig/mlpack-4.6.2
mlpack_lib := ${mlpack_dir}/build/lib
mlpack_include := ${mlpack_dir}/src

# Ensmallen
ensmallen_dir := /home/yorig/ensmallen-3.10.0
ensmallen_include := ${ensmallen_dir}/include

INCLUDE_DIRS = \
    -Iinclude \
    -I$(CPLEX_DIR)/cplex/include \
    -I$(CPLEX_DIR)/concert/include


LIB_DIRS = \
    -L$(CPLEX_DIR)/cplex/lib/x86-64_linux/static_pic \
    -L$(CPLEX_DIR)/concert/lib/x86-64_linux/static_pic

LIBS = -lilocplex -lcplex -lconcert -lm -lpthread -larmadillo \
       -lrt -lbz2 -ldl -lstdc++ -lopenblas -lgfortran -lz

ALLIANCE_INCLUDES = -I$(ensmallen_include) -I$(mlpack_include)
ALLIANCE_LIBDIRS  = -L$(mlpack_lib)

.PHONY: all clean mpi mpi_alliance
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

alliance: CXXFLAGS += $(ALLIANCE_CXXFLAGS)
alliance: INCLUDE_DIRS += $(ALLIANCE_INCLUDES)
alliance: LIB_DIRS += $(ALLIANCE_LIBDIRS)
alliance: all

mpi_alliance: CXX = $(CXX_MPI)
mpi_alliance: CXXFLAGS += $(MPI_FLAGS) $(ALLIANCE_CXXFLAGS)
mpi_alliance: INCLUDE_DIRS += $(ALLIANCE_INCLUDES)
mpi_alliance: LIB_DIRS += $(ALLIANCE_LIBDIRS)
mpi_alliance: all
