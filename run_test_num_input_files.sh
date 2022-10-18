#!/bin/bash
# use nullglob in case there are no matching files
shopt -s nullglob

# create an array with all the test inputs
files=( $(ls performance_tests/ | sed -r 's/(.[^;]*;)/ \1 /g' | tr " " "\n" | tr -d " " ) )

# prepend test folder name
files=( "${files[@]/#/'performance_tests/'}" )
for i in {10..200..10}; do
    python3 check.py signatures/sigs-both.txt ${files[@]:0:$i} > measurements/varies_input_files/"$i.txt"
done
