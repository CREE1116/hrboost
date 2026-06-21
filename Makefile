CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -march=native -fPIC -DNDEBUG
SRC      := src/hrboost.cpp src/capi_hrboost.cpp

# Detect OS
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    EXT     := dylib
    LDFLAGS := -dynamiclib -undefined dynamic_lookup
else
    EXT     := so
    LDFLAGS := -shared
endif

LIB := libhrboost.$(EXT)

all: $(LIB)

$(LIB): $(SRC) src/hrboost.h src/capi_hrboost.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -Isrc -o $@ $(SRC)

clean:
	rm -f $(LIB)

.PHONY: all clean