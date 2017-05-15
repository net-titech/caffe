#!/usr/bin/env sh
set -e

TOOLS=./build/tools

#finetune the trained cifar model with threshold pruning applied
$TOOLS/caffe train \
  --solver=examples/threshold_pruning/cifar10_finetune_solver.prototxt \
  --weights=examples/cifar10/cifar10_quick_iter_4000.caffemodel $@

