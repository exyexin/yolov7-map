#!/usr/bin/bash
for filename in $(ls *.txt); do
    mv $filename ${filename/RGB./}
done
