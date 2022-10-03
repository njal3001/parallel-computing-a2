# Makefile

NVCC        := /usr/local/cuda-11/bin/nvcc
COMMON_SRCS := src/common.cpp
CUDA_SRCS   := src/kernel.cu

OUTPUT      := scanner

.DEFAULT_TARGET: all

all: $(OUTPUT)

$(OUTPUT): $(CUDA_SRCS) $(COMMON_SRCS)
	$(NVCC) -std c++17 -O3 -lineinfo -dlto -arch=sm_70 -o $@ $^

clean:
	rm -f $(OUTPUT)
