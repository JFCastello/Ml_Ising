# Simple and robust Makefile
CXX      := g++
MPICXX   := mpicxx
CXXSTD   := c++17
CXXFLAGS := -std=$(CXXSTD) -O2 -Wall -Wextra -Iinclude
SRC_DIR  := src
OBJ_DIR  := build

# Possible MPI source files inside src/
MPI_CANDIDATES := $(SRC_DIR)/All_Configurations.cpp $(SRC_DIR)/All_configurations.cpp
MPI_SRC := $(wildcard $(MPI_CANDIDATES))

# main.cpp at the root (if it exists)
ROOT_MAIN := $(wildcard main.cpp)

# Normal sources: main.cpp (if present) + all src/*.cpp except MPI sources
NORMAL_SRCS := $(ROOT_MAIN) $(filter-out $(MPI_SRC),$(wildcard $(SRC_DIR)/*.cpp))

# Objects: build/<basename>.o (notdir avoids path in object name)
OBJS := $(addprefix $(OBJ_DIR)/,$(patsubst %.cpp,%.o,$(notdir $(NORMAL_SRCS))))

# Final executable (at root, because main.cpp lives there)
TARGET := main

# MPI executables with _mpi suffix, placed in src/
MPI_PROGS := $(patsubst $(SRC_DIR)/%.cpp,$(SRC_DIR)/%_mpi,$(MPI_SRC))

.PHONY: all mpi clean

all: $(TARGET)
	@echo "Build complete."

# Link the main executable from all object files
$(TARGET): $(OBJS)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Generic rule to compile any build/<basename>.o
# Looks for the matching .cpp first in src/, then at the root.
$(OBJ_DIR)/%.o:
	@mkdir -p $(OBJ_DIR)
	@echo "Compiling $@"
	@if [ -f "$(SRC_DIR)/$*.cpp" ]; then SRC="$(SRC_DIR)/$*.cpp"; \
	elif [ -f "$*.cpp" ]; then SRC="$*.cpp"; \
	else echo "ERROR: source not found for $*"; exit 1; fi; \
	$(CXX) $(CXXFLAGS) -c $$SRC -o $@

# Target to build all MPI sources (if any exist)
mpi: $(MPI_PROGS)
	@echo "MPI build complete."

$(SRC_DIR)/%_mpi: $(SRC_DIR)/%.cpp
	@echo "Compiling (MPI) $< -> $@"
	$(MPICXX) -std=$(CXXSTD) -O2 -Wall -Wextra -Iinclude -o $@ $<

clean:
	@echo "Cleaning build artifacts, executables, and .txt data files..."
	-rm -rf $(OBJ_DIR)
	-rm -f $(TARGET) $(MPI_PROGS)
	# delete generated data .txt files (but not requirements.txt or docs)
	-find ./data -type f -name '*.txt' -print -exec rm -f {} +
	@echo "Clean complete."
