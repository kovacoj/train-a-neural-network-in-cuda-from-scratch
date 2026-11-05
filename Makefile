NVCC ?= nvcc
TARGET := cuda-mlp
SOURCES := src/main.cpp src/data.cpp src/kernels.cu src/mlp.cu src/benchmark.cu
HEADERS := $(wildcard include/*.h)
NVCCFLAGS ?= -O2 -std=c++17
CPPFLAGS += -Iinclude

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(SOURCES) -o $@

clean:
	rm -f $(TARGET)
