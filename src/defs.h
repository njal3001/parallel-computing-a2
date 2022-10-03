// common definitions

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstdarg>

#include <vector>
#include <string>
#include <utility>

struct Signature
{
	std::string name;
	char* data;
	size_t size;
};

struct InputFile
{
	std::string name;
	uint8_t* data;
	size_t size;
};

uint64_t get_nanoseconds();

InputFile readInputFile(const char* filename);
std::vector<Signature> readSignatures(const char* signature_db, void** ret_ptr, size_t* ret_size);

void runScanner(std::vector<Signature>& signatures, std::vector<InputFile>& input_files);

#if defined(__CUDACC__)

#define check_cuda_error(ans) gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if(code != cudaSuccess)
	{
		fprintf(stderr, "cuda assertion failed (%s:%d): %s\n", file, line, cudaGetErrorString(code));
		if(abort)
			exit(code);
	}
}
#endif


[[noreturn]] inline void error_and_exit(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);

	vfprintf(stderr, fmt, ap);
	va_end(ap);

	std::exit(1);
}
