"""
Takes as arguments:
1. the path to a model definition prototxt file
2. the path to caffemodel file
3. prunning ratio: what percentage of weights need to be pruned from convolution and fully-connected layers (eg: if this vaue is 30, 30% smallest weights will be set to zero, leaving only 70% of the weights)
4. the path to output prototxt file

"""

import numpy as np
import sys

caffe_root = '../../'  ## this file should be run from {caffe_root} (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

if len(sys.argv) != 5:
  print "please input the correct number of arguments"

model_def = sys.argv[1]
model_weights = sys.argv[2]
prune_threshold = float(sys.argv[3])
output_file = sys.argv[4]

if prune_threshold <= 0 or prune_threshold >=100:
  print "pruning ratio should be between 0 and 100"
  quit()

caffe.set_mode_cpu()

net = caffe.Net(model_def, model_weights, caffe.TEST) 
pruned_net = caffe_pb2.NetParameter()

with open(model_def) as mf:
  txtf.Merge(mf.read(), pruned_net)

lnames = [l.name for l in pruned_net.layer]

for lname in lnames:
  layer_type = net.layer_dict[lname].type
  if layer_type == 'Convolution' or layer_type == 'InnerProduct':
    weights = np.abs(net.params[lname][0].data)
    threshold = np.percentile(weights, 100.0-prune_threshold)
    pruned_net.layer[lnames.index(lname)].pruning_param.threshold = threshold

with open(output_file, 'w') as f:
  f.write(str(pruned_net))

