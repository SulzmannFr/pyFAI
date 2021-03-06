#!/bin/sh
rm -f src/histogram.c
rm -f src/histogram.so
rm -f src/histogram.pyx
echo "OpenMP code"
cp src/histogram_omp.pyx src/histogram.pyx
cython -a src/histogram.pyx --fast-fail
cp src/histogram.c src/histogram_omp.c
cp src/histogram.html src/histogram_omp.html

rm -f src/histogram.c
rm -f src/histogram.so
rm -f src/histogram.pyx
echo "No OpenMP code"
cp src/histogram_nomp.pyx src/histogram.pyx
cython -a src/histogram.pyx --fast-fail
cp src/histogram.c src/histogram_nomp.c
cp src/histogram.html src/histogram_nomp.html

rm -f src/histogram.c
rm -f src/histogram.so
rm -f src/histogram.pyx


