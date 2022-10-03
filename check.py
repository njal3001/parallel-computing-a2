#!/usr/bin/env python3

import os
import sys
import time
import random
import hashlib
import itertools
import subprocess

CACHE_FILE = "cached-results.txt"

SCANNER_PAR = "scanner"
SCANNER_SEQ = "scanner-seq"

from_iterable = itertools.chain.from_iterable

def kv_list_to_dict(kvl):
	d = dict()
	for kv in kvl:
		d.setdefault(kv[0], set()).add(kv[1])
	return d

def print_time(ns):
	ms = ns / 1000000
	if ms > 1000:
		return f"{ms / 1000:.3f}s"
	else:
		return f"{ms / 1000:.1f}ms"


def main(sig_file, inputs):
	cache_header = f"{sig_file}_{'_'.join(sorted(inputs))}"
	cache_hash = hashlib.sha256(cache_header.encode()).hexdigest()

	print(f"cache hash: {cache_hash}")

	cached = False
	sequential_time_ns = 0

	if os.path.exists(CACHE_FILE):
		aa = open(CACHE_FILE, "r").read().splitlines()
		if len(aa) > 1:
			first_line = aa[0].strip()
			if first_line == cache_hash:
				cached = True
				sequential_time_ns = int(aa[1].strip())
				parallel_time_ns = int(aa[2].strip())

	cached_lines = []
	if cached:
		print(f"cached results up to date")
		cached_lines = list(map(lambda x: x.strip(), open(CACHE_FILE, "r").read().splitlines()[3:]))
	else:
		print(f"stale cache, rerunning sequential implementation")
		print(f"this may take a while...")

		if not os.path.exists(SCANNER_SEQ):
			print(f"sequential implementation ('{SCANNER_SEQ}') does not exist, aborting!")
			sys.exit(1)

		env = os.environ
		env["PARALLEL_HAX"] = "1"
		env["PRINT_PROGRESS"] = "1"

		# run our sequential thing in parallel, but don't tell them
		start_ns = time.monotonic_ns()
		p = subprocess.run([ f"./{SCANNER_SEQ}", sig_file, *inputs ], stdout=subprocess.PIPE, env=env)
		parallel_time_ns = time.monotonic_ns() - start_ns

		if p.returncode != 0:
			print(f"failed!")
			sys.exit(1)

		cached_lines = list(map(lambda x: x.strip(), p.stdout.decode().splitlines()))
		if not cached_lines[-1].startswith("ELAPSED"):
			print(f"failed!")
			sys.exit(1)

		sequential_time_ns = int(cached_lines[-1].strip().split(' ')[1])
		cached_lines = cached_lines[:-1]
		open(CACHE_FILE, "w").write('\n'.join([
			cache_hash,
			str(sequential_time_ns),
			str(parallel_time_ns),
			*cached_lines
		]))

	expected_output = kv_list_to_dict(map(lambda l: (l.split(':')[0].strip(), l.split(':')[1].strip()),
		cached_lines))

	print("")
	print(f"running your implementation...")
	print(f"ideally, this should not take so long (:")
	print(f"time to beat: {print_time(sequential_time_ns)}")

	start_ns = time.monotonic_ns()
	p = subprocess.run([ f"./{SCANNER_PAR}", sig_file, *inputs ], stdout=subprocess.PIPE)
	cuda_runtime_ns = time.monotonic_ns() - start_ns

	if p.returncode != 0:
		print(f"failed!")
		sys.exit(1)

	actual_lines = list(map(lambda x: x.strip(), p.stdout.decode().splitlines()))
	actual_output = kv_list_to_dict(map(lambda l: (l.split(':')[0].strip(), l.split(':')[1].strip()),
		actual_lines))

	false_negatives = kv_list_to_dict(from_iterable(map(lambda f: [(f, a) for a in expected_output[f]],
		set(expected_output.keys()) - set(actual_output.keys()))))

	false_positives = kv_list_to_dict(from_iterable(map(lambda f: [(f, a) for a in actual_output[f] if (f not in expected_output) or (a not in expected_output[f])],
		actual_output.keys())))

	def print_dict(d):
		for k in sorted(d.keys()):
			print(f"  {k}:")
			for v in sorted(d[k]):
				print(f"   * {v}")

	num_fp = sum(map(len, false_positives.values()))
	num_fn = sum(map(len, false_negatives.values()))

	print("-------------------------")

	print("false positives:")
	if num_fp > 0:
		print_dict(false_positives)
	else:
		print("none")
	print("-------------------------")

	print("false negatives:")
	if num_fn > 0:
		print_dict(false_negatives)
	else:
		print("none")
	print("-------------------------")

	num_correct = len(list(from_iterable(map(lambda f: [(f, a) for a in actual_output[f] if a in expected_output[f]],
		set.intersection(set(expected_output.keys()), set(actual_output.keys()))))))

	print("")
	print("summary:")
	print(f"  true pos:    {num_correct}")
	print(f"  false pos:   {num_fp}")
	print(f"  false neg:   {num_fn}")

	BETA = 15.0

	beta_sq = BETA*BETA
	foo = 1 + beta_sq

	print("")


	print(f"time taken: {print_time(cuda_runtime_ns)} vs {print_time(sequential_time_ns)}")
	print(f"speedup: {sequential_time_ns / cuda_runtime_ns:.3f}x")

	f_beta_score = (foo*num_correct) / (foo*num_correct + beta_sq*num_fn + num_fp)
	print(f"F-beta = {f_beta_score:.4f}")

	print(f"")
	print(f"parallel speedup (for bonus): {parallel_time_ns / cuda_runtime_ns:.3f}x")
	print(f"")


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print(f"usage: ./{sys.argv[0]} <signature_db> [input_file]...")
		sys.exit(0)

	main(sys.argv[1], sys.argv[2:])
