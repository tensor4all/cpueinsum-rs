#!/bin/bash
# cpueinsum-rs#80 measurement: per thread count, native / base / cand, two rounds.
# Every cell: median of 11 warm calls after 1000 ms priming per workload.
set -u
B=${BENCH_DIR:?}
W=$B/workloads.txt
NATIVE=$HOME/work/symtn-rs/target/release/examples/native80
BASE=$HOME/work/cpueinsum-rs-base/target/release/examples/gp80
CAND=$HOME/work/cpueinsum-rs-cand/target/release/examples/gp80
unset RAYON_NUM_THREADS OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
T=$1; CORES=$2; OUT=$B/raw-t$T.tsv
: > $OUT
for round in 1 2; do
  for arm in native base cand; do
    case $arm in native) BIN=$NATIVE;; base) BIN=$BASE;; cand) BIN=$CAND;; esac
    echo "# round=$round arm=$arm T=$T cores=$CORES load=$(cut -d' ' -f1 /proc/loadavg) $(date -Is)" >> $OUT
    taskset -c $CORES $BIN $W $T 1000 11 | sed "s/^/$round\t$arm\t/" >> $OUT
  done
done
echo "# end load=$(cut -d' ' -f1 /proc/loadavg) $(date -Is)" >> $OUT
