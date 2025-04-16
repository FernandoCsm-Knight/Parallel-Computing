# ============================================================
# 1. Configuration Section
# ============================================================
MPI ?= 0

ifeq ($(MPI),1)
    CXX = mpic++
    CXXFLAGS = -fopenmp -std=c++23 -Wall -Iinclude
    $(info Building with MPI (mpic++), using OpenMP)
else
    CXX = g++
    CXXFLAGS = -fopenmp -std=c++23 -Wall -Iinclude
    $(info Building with clang++ (no MPI), using OpenMP)
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
SRC_FILES    := $(shell find $(SRC_DIR) -name '*.cpp') $(wildcard ./*.cpp)
OBJ_FILES    := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(SRC_FILES)))
BIN_FILE     = $(BIN_DIR)/program

# ============================================================
# 4. Default Rule
# ============================================================
all: build

# ============================================================
# 5. Build the Project
# ============================================================
build: $(BIN_FILE)

$(BIN_FILE): $(OBJ_FILES) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(addprefix $(OBJ_DIR)/, $(notdir $(SRC_FILES:.cpp=.o)))

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ============================================================
# 6. Directory Creation
# ============================================================
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# ============================================================
# 7. Run, Clean, Rerun
# ============================================================
run: build
ifeq ($(MPI),0)
	./$(BIN_FILE)
else
	mpirun -np $(MPI) ./$(BIN_FILE)
endif

clean:
	@if [ -d "$(LIB_DIR)" ]; then rm -rf $(LIB_DIR); fi

time: clean build
ifeq ($(MPI),0)
	@echo Timing...
	/usr/bin/time ./$(BIN_FILE)
else
	@echo Timing...
	/usr/bin/time  mpirun -np $(MPI) ./$(BIN_FILE)
endif

rerun: clean build run

valgrind:
	@echo "Valgrind não disponível nativamente no Windows. Use WSL."

# ============================================================
# 8. Phony Targets
# ============================================================
.PHONY: all build run clean rerun valgrind time
