CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -march=native -ffast-math -fPIC -DNDEBUG
SRC      := src/brstboost.cpp src/capi.cpp

# Detect OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    EXT     := dylib
    LDFLAGS := -dynamiclib -undefined dynamic_lookup
else
    EXT     := so
    LDFLAGS := -shared
endif

LIB := libbrstboost.$(EXT)

all: $(LIB)

$(LIB): $(SRC) src/brstboost.h src/capi.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -Isrc -o $@ $(SRC)

clean:
	rm -f $(LIB)

.PHONY: all clean