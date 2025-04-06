# ============================================================
# 1. Configuration Section
# ============================================================
# If MPI=1, use mpic++ (OpenMPI’s C++ compiler wrapper).
# Otherwise, default to g++.
MPI ?= 0

ifeq ($(shell test $(MPI) -gt 0 && echo true), true)
    CXX = mpic++
    # Adicione quaisquer flags específicas do MPI, se necessário.
    CXXFLAGS = -fopenmp -std=c++23 -Wall -Iinclude
    $(info Building with MPI (mpic++), using OpenMP)
else
    CXX = g++
    CXXFLAGS = -fopenmp -std=c++23 -Wall -Iinclude
    $(info Building with g++ (no MPI), using OpenMP)
endif

# ============================================================
# 2. Directories
# ============================================================
SRC_DIR      = src
LIB_DIR      = lib
OBJ_DIR      = $(LIB_DIR)/obj
BIN_DIR      = $(LIB_DIR)/bin
INCLUDE_DIR  = inc

# ============================================================
# 3. Source and Object Files
# ============================================================
# Find all .cpp files in the current directory and in src/**
SRC_FILES    := $(wildcard ./*.cpp) $(wildcard $(SRC_DIR)/**/*.cpp)

# Convert the .cpp paths to .o paths under OBJ_DIR
OBJ_FILES    := $(patsubst ./%.cpp,      $(OBJ_DIR)/%.o, \
                 $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES)))

# Unique directories for object files (in case of subfolders)
OBJ_DIRS     := $(sort $(dir $(OBJ_FILES)))

# Output binary
BIN_FILE     = $(BIN_DIR)/program

# ============================================================
# 4. Default Rule
# ============================================================
all: build

# ============================================================
# 5. Build the Project
# ============================================================
build: $(BIN_FILE)

# Compile the final executable
$(BIN_FILE): $(OBJ_FILES) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile .cpp to .o (keeping directory structure)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ============================================================
# 6. Directory Creation
# ============================================================
$(BIN_DIR) $(OBJ_DIR):
	@mkdir -p $@

$(OBJ_DIRS):
	@mkdir -p $@

# ============================================================
# 7. Run, Clean, Rerun
# ============================================================
run: build
	@if [ $(MPI) -eq 0 ]; then \
		$(BIN_FILE); \
	else \
		mpirun -np $(MPI) $(BIN_FILE); \
	fi

clean:
	rm -rf $(LIB_DIR)

time: build 
	@if [ $(MPI) -eq 0 ]; then \
		time -v $(BIN_FILE); \
	else \
		time -v mpirun -np $(MPI) $(BIN_FILE); \
	fi

rerun: clean build run

valgrind: build
	@if [ $(MPI) -eq 0 ]; then \
		valgrind --leak-check=full --track-origins=yes $(BIN_FILE); \
	else \
		mpirun -np $(MPI) valgrind --leak-check=full --track-origins=yes $(BIN_FILE); \
	fi

# ============================================================
# 8. Phony Targets
# ============================================================
.PHONY: all build run clean rerun
