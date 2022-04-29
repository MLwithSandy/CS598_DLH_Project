from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

d = np.loadtxt(args.filename).mean(axis=0)

print(d[0])
print(d[1])
print(d[2])
print(d[3] / 319698 * 1000)
print(d[4] / 79925 * 1000)
