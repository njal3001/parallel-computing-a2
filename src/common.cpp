// common.cpp
// common code for input parsing

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cerrno>

#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <utility>
#include <filesystem>
#include <string_view>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "defs.h"

namespace stdfs = std::filesystem;

uint64_t get_nanoseconds()
{
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return ((uint64_t) tp.tv_nsec + (uint64_t) tp.tv_sec * 1000000000ll);
}

static std::pair<void*, size_t> mmap_file(const char* filename)
{
	int fd = open(filename, O_RDONLY);
	if(fd < 0)
		error_and_exit("open failed: %s (%d)", strerror(errno), errno);

	auto file_size = stdfs::file_size(filename);
	void* ptr = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
	if(ptr == MAP_FAILED)
		error_and_exit("mmap failed: %s (%d)", strerror(errno), errno);

	// mmap will keep the file alive until we munmap it.
	close(fd);

	return { ptr, file_size };
}

std::vector<Signature> readSignatures(const char* signature_db, void** ret_ptr, size_t* ret_size)
{
	auto [ptr, file_size] = mmap_file(signature_db);

	*ret_ptr = ptr;
	*ret_size = file_size;

	std::vector<Signature> signatures {};
	auto contents = std::string_view(static_cast<const char*>(ptr), file_size);
	while(contents.size() > 0)
	{
		auto eol = contents.find_first_of("\r\n");
		auto line = contents.substr(0, eol);

		contents.remove_prefix(line.size());

		while(contents.size() > 0 && (contents[0] == '\r' || contents[0] == '\n'))
			contents.remove_prefix(1);

		if(line.size() > 0 && line.front() == '#')
			continue;

		auto i = line.find(':');
		auto name = line.substr(0, i);
		auto hex  = line.substr(i + 1);

		Signature sig {};
		sig.name = name;
		sig.data = (char*) hex.data();
		sig.size = hex.size();

		signatures.push_back(std::move(sig));
	}

	return signatures;
}

InputFile readInputFile(const char* filename)
{
	auto [ptr, file_size] = mmap_file(filename);

	InputFile file {};
	file.name = filename;
	file.data = static_cast<uint8_t*>(ptr);
	file.size = file_size;

	return file;
}

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		printf("usage: %s <signature database> [input files]...\n", argv[0]);
		return 0;
	}
	else if(not stdfs::exists(argv[1]))
	{
		error_and_exit("signature database '%s' does not exist\n", argv[1]);
	}

	std::vector<InputFile> input_files {};
	for(int i = 2; i < argc; i++)
	{
		if(not stdfs::exists(argv[i]))
			error_and_exit("input file '%s' does not exist\n", argv[i]);

		input_files.push_back(readInputFile(argv[i]));
	}

	void* signature_db_buf = 0;
	size_t signature_db_size = 0;
	auto signatures = readSignatures(argv[1], &signature_db_buf, &signature_db_size);

	fprintf(stderr, "loaded %zu signature(s), %zu input file(s)\n",
		signatures.size(), input_files.size());

	runScanner(signatures, input_files);

	// cleanup memory
	munmap(signature_db_buf, signature_db_size);
}
