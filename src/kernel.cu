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

#include <vector>

#include "defs.h"

__global__ void matchFile(const uint8_t* file_data, size_t file_len, const char* signature, size_t len)
{
	// TODO: your code!
}

void runScanner(std::vector<Signature>& signatures, std::vector<InputFile>& inputs)
{
	{
		cudaDeviceProp prop;
		check_cuda_error(cudaGetDeviceProperties(&prop, 0));

		fprintf(stderr, "cuda stats:\n");
		fprintf(stderr, "  # of SMs: %d\n", prop.multiProcessorCount);
		fprintf(stderr, "  global memory: %.2f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
		fprintf(stderr, "  shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
		fprintf(stderr, "  constant memory: %zu bytes\n", prop.totalConstMem);
	}

	/*
		Here, we are creating one stream per file just for demonstration purposes;
		you should change this to fit your own algorithm and/or implementation.
	*/
	std::vector<cudaStream_t> streams {};
	streams.resize(inputs.size());

	std::vector<uint8_t*> file_bufs {};
	for(size_t i = 0; i < inputs.size(); i++)
	{
		cudaStreamCreate(&streams[i]);

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
		cudaMemcpy(ptr, signatures[i].data, signatures[i].size, cudaMemcpyHostToDevice);
		sig_bufs.push_back(ptr);
	}

	for(size_t file_idx = 0; file_idx < inputs.size(); file_idx++)
	{
		// asynchronously copy the file contents from host memory
		// (the `inputs`) to device memory (file_bufs, which we allocated above)
		cudaMemcpyAsync(file_bufs[file_idx], inputs[file_idx].data, inputs[file_idx].size,
			cudaMemcpyHostToDevice, streams[file_idx]);    // pass in the stream here to do this async

		for(size_t sig_idx = 0; sig_idx < signatures.size(); sig_idx++)
		{
			// launch the kernel!
			// your job: figure out the optimal dimensions

			/*
				This launch happens asynchronously. This means that the CUDA driver returns control
				to our code immediately, without waiting for the kernel to finish. We can then
				run another iteration of this loop to launch more kernels.

				Each operation on a given stream is serialised; in our example here, we launch
				all signatures on the same stream for a file, meaning that, in practice, we get
				a maximum of NUM_INPUTS kernels running concurrently.

				Of course, the hardware can have lower limits; on Compute Capability 8.0, at most
				128 kernels can run concurrently --- subject to resource constraints. This means
				you should *definitely* be doing more work per kernel than in our example!
			*/
			matchFile<<<1, 1, /* shared memory per block: */ 0, streams[file_idx]>>>(
				file_bufs[file_idx], inputs[file_idx].size,
				sig_bufs[sig_idx], signatures[sig_idx].size);


			// example output printing. don't forget to change this!
			printf("%s: %s\n", inputs[file_idx].name.c_str(), signatures[sig_idx].name.c_str());
		}
	}


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
