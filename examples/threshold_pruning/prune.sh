#!/bin/bash

prototxt=$1
model=$2
output=$3
solver="examples/threshold_pruning/alexnet/solver.prototxt"

# prunes from 1-50% of the network weights
for i in `seq 50 -1 1`;
do
    outfile="${output}/${i}/train_val.prototxt"
    newsolver="${output}/${i}/solver.prototxt"
    mkdir -p examples/threshold_pruning/alexnet/$i/log
    mkdir -p examples/threshold_pruning/alexnet/$i/snapshots
    rep='s/net: "(.*)\/train_val.prototxt"/net: "\1\/'$i'\/train_val.prototxt"/g'
    rep2='s/snapshot_prefix: "(.*)\/caffe_alexnet_train"/snapshot_prefix: "\1\/'$i'\/snapshots\/caffe_alexnet_train"/g'
    sed -r "$rep;$rep2" $solver > $newsolver
    python examples/threshold_pruning/prune_net.py $1 $2 $i $outfile
    ./build/tools/caffe train --solver=$newsolver --weights=$model -log_dir=examples/threshold_pruning/alexnet/$i/log
done
