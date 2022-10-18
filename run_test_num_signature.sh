#!/bin/bash

# Extract signatures
for i in {100..2000..100}; do
    cat signatures/sigs-both.txt | head -n $i > signatures/$i.txt
done

# Run evaluation
for i in {100..2000..100}; do
    python3 check.py signatures/$i.txt performance_tests/* > measurements/varies_num_signatures/$i.txt
done

# Cleanup signature files
for i in {100..2000..100}; do
    rm signatures/$i.txt
done