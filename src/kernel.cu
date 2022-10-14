/*
        CS3210 Assignment 2
        CUDA Virus Scanning

        Most of your CUDA code should go in here.

        Feel free to change any code in the skeleton, as long as you conform
        to the input and output formats specified in the assignment pdf.

        If you rename this file or add new files, remember to modify the
        Makefile! Just make sure (haha) that the default target still builds
        your program, and you don't rename the program (`scanner`).

        The skeleton demonstrates how asnychronous kernel launches can be
        done; it is up to you to decide (and implement!) the parallelisation
        paradigm for the kernel. The provided implementation is not great,
        since it launches one kernel per file+signature combination (a lot!).
        You should try to do more work per kernel in your implementation.

        You can launch as many kernels as you want; if any preprocessing is
        needed for your algorithm of choice, you can also do that on the GPU
        by running different kernels.

        'defs.h' contains the definitions of the structs containing the input
        and signature data parsed by the provided skeleton code; there should
        be no need to change it, but you can if you want to.

        'common.cpp' contains the aforementioned parsing for the input files.
        The input files are already efficiently read with mmap(), so there
        should be little to no gain trying to optimise that portion of the
        skeleton.

        Remember: print any debugging statements to STDERR!
*/

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "defs.h"

__device__ int matchSignature(const uint8_t *data, const char *signature, size_t len)
{
	const char *hex_table = "0123456789abcdef";
	for (size_t i = 0; i < len / 2; i++)
	{
		uint8_t hl = data[i] & 0xF;
		uint8_t hh = (data[i] >> 4) & 0xF;

		size_t j = 2 * i;
		// printf("Comparing %c%c to %c%c\n", signature[j], signature[j+1], hex_table[hh], hex_table[hl]);
		if (signature[j] != '?' && hex_table[hh] != signature[j])
		{
			return 0;
		}		
		if (signature[j + 1] != '?' && hex_table[hl] != signature[j + 1])
		{
			return 0;
		}		
	}

	/*
	printf("Matched %s with ", signature);
	for (size_t i = 0; i < len / 2; i++)
	{
		uint8_t hl = data[i] & 0xF;
		uint8_t hh = (data[i] >> 4) & 0xF;

		printf("%c%c", hex_table[hh], hex_table[hl]);
	}
	printf("\n");
	*/
	return 1;
}

__global__ void matchFile(const uint8_t* file_data, size_t file_len, const char* signature, size_t len, uint8_t *result)
{
	size_t num_threads_per_block = blockDim.x;
	size_t num_blocks = gridDim.x;
	size_t num_threads = num_threads_per_block * num_blocks;
	size_t n_max = file_len - len / 2 + 1;
	size_t workload = (n_max - 1) / num_threads + 1;
	// printf("Launched thread: %d, block: %d\n", threadIdx.x, blockIdx.x);
	
	size_t n_start = (blockIdx.x * num_threads_per_block + threadIdx.x) * workload;
	size_t n_end = min(n_start + workload, n_max);

	// printf("Checking %s\n", signature);
	// printf("File size: %lu, Signature size: %lu, Iterations: %lu, Workload: %lu, Offset: %lu, End: %lu\n",
	//	file_len, len, n_max, workload, n_start, n_end);
	for (size_t n = n_start; n < n_end; n++)
	{
		if (matchSignature(file_data + n, signature, len))
		{
			*result = 1;
			return;
		}
	}
}

void runScanner(std::vector<Signature>& signatures, std::vector<InputFile>& inputs)
{
	cudaDeviceProp prop;
	check_cuda_error(cudaGetDeviceProperties(&prop, 0));

	fprintf(stderr, "cuda stats:\n");
	fprintf(stderr, "  # of SMs: %d\n", prop.multiProcessorCount);
	fprintf(stderr, "  global memory: %.2f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
	fprintf(stderr, "  shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
	fprintf(stderr, "  constant memory: %zu bytes\n", prop.totalConstMem);

	/*
		Here, we are creating one stream per file just for demonstration purposes;
		you should change this to fit your own algorithm and/or implementation.
	*/
	std::vector<cudaStream_t> streams {};
	streams.resize(inputs.size());

	std::vector<uint8_t*> file_bufs {};
	for(size_t i = 0; i < inputs.size(); i++)
	{
		check_cuda_error(cudaStreamCreate(&streams[i]));

		// allocate memory on the device for the file
		uint8_t* ptr = 0;
		check_cuda_error(cudaMalloc(&ptr, inputs[i].size));
		file_bufs.push_back(ptr);
	}

	// allocate memory for the signatures
	std::vector<char*> sig_bufs {};
	for(size_t i = 0; i < signatures.size(); i++)
	{
		char* ptr = 0;
		check_cuda_error(cudaMalloc(&ptr, signatures[i].size));
		check_cuda_error(cudaMemcpy(ptr, signatures[i].data, signatures[i].size,
			cudaMemcpyHostToDevice));
		sig_bufs.push_back(ptr);
	}
	
	uint8_t *dresult;
	size_t result_size = inputs.size() * signatures.size() * sizeof(uint8_t);
	check_cuda_error(cudaMalloc(&dresult, result_size));
	// printf("Result size: %lu\n", result_size);

	cudaMemset(dresult, 0, result_size);

	for(size_t file_idx = 0; file_idx < inputs.size(); file_idx++)
	{
		// printf("File %lu\n", file_idx);
		// asynchronously copy the file contents from host memory
		// (the `inputs`) to device memory (file_bufs, which we allocated above)
		check_cuda_error(cudaMemcpyAsync(file_bufs[file_idx], inputs[file_idx].data,
			inputs[file_idx].size, cudaMemcpyHostToDevice, streams[file_idx]));

		for(size_t sig_idx = 0; sig_idx < signatures.size(); sig_idx++)
		{
			// launch the kernel!
			// your job: figure out the optimal dimensions

			/*
				This launch happen asynchronously. This means that the CUDA driver returns control
				to our code immediately, without waiting for the kernel to finish. We can then
				run another iteration of this loop to launch more kernels.

				Each operation on a given stream is serialised; in our example here, we launch
				all signatures on the same stream for a file, meaning that, in practice, we get
				a maximum of NUM_INPUTS kernels running concurrently.

				Of course, the hardware can have lower limits; on Compute Capability 8.0, at most
				128 kernels can run concurrently --- subject to resource constraints. This means
				you should *definitely* be doing more work per kernel than in our example!
			*/
			matchFile<<<32, 32, /* shared memory per block: */ 0, streams[file_idx]>>>(
				file_bufs[file_idx], inputs[file_idx].size,
				sig_bufs[sig_idx], signatures[sig_idx].size,
				dresult + file_idx * signatures.size() + sig_idx);

			// printf("%s: %s\n", inputs[file_idx].name.c_str(), signatures[sig_idx].name.c_str());
		}
	}

	uint8_t *result = (uint8_t*)malloc(result_size);
	check_cuda_error(cudaMemcpy(result, dresult, result_size, cudaMemcpyDeviceToHost));

	for(size_t file_idx = 0; file_idx < inputs.size(); file_idx++)
	{
		for(size_t sig_idx = 0; sig_idx < signatures.size(); sig_idx++)
		{
			if (result[file_idx * signatures.size() + sig_idx])
			{
				printf("%s: %s\n", inputs[file_idx].name.c_str(),
					signatures[sig_idx].name.c_str());
			}	
		}
	}

	cudaFree(dresult);
	free(result);

	// free the device memory, though this is not strictly necessary
	// (the CUDA driver will clean up when your program exits)
	for(auto buf : file_bufs)
		cudaFree(buf);

	for(auto buf : sig_bufs)
		cudaFree(buf);

	// clean up streams (again, not strictly necessary)
	for(auto& s : streams)
		cudaStreamDestroy(s);
}
