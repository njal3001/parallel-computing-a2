#!/usr/bin/python3

import os
import sys
import random
import itertools
from typing import Iterable, Tuple

SEED = os.urandom(32)
MIN_FILE_SIZE = 4096
MAX_FILE_SIZE = 8 * 1024 * 1024

def random_bytes(n: int) -> bytes:
	if sys.version_info[1] >= 9:        # random.randbytes only in 3.9+
		return random.randbytes(n)
	else:
		return bytes([ random.getrandbits(8) for _ in range(0, n) ])



def parse_signatures(lines: Iterable[str]) -> Iterable[Tuple[bytes, str]]:
	for line in lines:
		kv = line.strip().split(':')
		assert len(kv) == 2

		if '?' not in kv[1]:
			yield bytes.fromhex(kv[1]), kv[0]
		else:
			n = 0
			# https://stackoverflow.com/questions/69792413
			chars = random.choices("0123456789abcdef", k=7)
			for p in map(iter, itertools.product(chars, repeat=kv[1].count('?'))):
				if n < 1024:
					n += 1
					yield bytes.fromhex(''.join(c if c != '?' else next(p) for c in kv[1])), kv[0]
				else:
					break


def human_readable_size(sz: int) -> str:
	if sz < 1024:
		return f"{sz} bytes"
	elif sz < 1024*1024:
		return f"{sz / 1024:.2f} kB"
	else:
		return f"{sz / (1024*1024):.2f} MB"


def gen_file_size(total_virus_size: int) -> int:
	min_size = max(MIN_FILE_SIZE, total_virus_size + 1024)

	# https://observablehq.com/@herbps10/beta-distribution
	return max(min_size, int(MAX_FILE_SIZE * random.betavariate(1.1, 10.5)))



def generate_benign(idx: int, sigs: Iterable[Tuple[bytes, str]]) -> Tuple[str, bytes]:
	split_into = int(2.3 * len(sigs))

	pieces = list(map(lambda s: [s[0][i::split_into] for i in range(split_into)], sigs))
	pieces = list(itertools.chain.from_iterable(pieces))
	random.shuffle(pieces)

	total_virus_size = sum(map(len, pieces))
	file_size = gen_file_size(total_virus_size)

	file = bytearray()
	remaining_virus_size = total_virus_size

	p = 0
	just_added = False
	while p < len(pieces):
		remaining = file_size - remaining_virus_size - len(file)
		if random.random() < 0.5 or just_added:
			just_added = False
			file += random_bytes(random.randrange(0, remaining))
		else :
			just_added = True
			remaining_virus_size -= len(pieces[p])
			file += pieces[p]
			p += 1

	if len(file) < file_size:
		file += random_bytes(file_size - len(file))

	return f"benign-{idx:04}", file



def generate_virus(idx: int, sigs: Iterable[Tuple[bytes, str]]) -> Tuple[str, bytes]:
	total_virus_size = sum(map(lambda s: len(s[0]), sigs))
	file_size = gen_file_size(total_virus_size)

	file = bytearray()
	remaining_virus_size = total_virus_size

	v = 0
	while v < len(sigs):
		remaining = file_size - remaining_virus_size - len(file)
		if random.random() < 0.5:
			file += random_bytes(random.randrange(0, remaining))
		else:
			file += sigs[v][0]
			remaining_virus_size -= len(sigs[v][0])
			v += 1

	if len(file) < file_size:
		file += random_bytes(file_size - len(file))

	return (f"virus-{idx:04}-" + '+'.join(map(lambda s: s[1], sigs))), file



def main(sig_file: str, output_dir: str, num_files: int, virus_chance: float, max_viruses: int):
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	random.seed(SEED)
	print(f"seed = {SEED.hex()}")

	print(f"num files:    {num_files}")
	print(f"virus chance: {100 * virus_chance:.1f}%")
	print(f"max per file: {max_viruses}")

	with open(sig_file, "r") as f:
		all_sigs = list(parse_signatures(f))

	virus_idx = 1
	benign_idx = 1
	for i in range(0, num_files):
		is_virus = random.random() < virus_chance

		virus_count = max_viruses
		viruses = random.choices(all_sigs, k=virus_count)

		if is_virus:
			filename, file = generate_virus(virus_idx, viruses)
			virus_idx += 1
		else:
			filename, file = generate_benign(benign_idx, viruses)
			benign_idx += 1

		print(f"{human_readable_size(len(file))} - {filename}")
		open(f"{output_dir}/{filename}.in", "wb").write(file)



if __name__ == "__main__":
	if len(sys.argv) < 5:
		print(f"usage: ./gen_tests.py <signatures.txt> <output_dir> <num_files> <virus_chance> <max_viruses_per_file>")
		sys.exit(0)

	main(sys.argv[1], sys.argv[2], int(sys.argv[3]), min(1, max(-1, float(sys.argv[4]))), int(sys.argv[5]))
