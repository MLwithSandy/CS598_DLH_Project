from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import random

from util import read_diagnosis, read_notes


diagnosis, icd9_lookup = read_diagnosis()
texts, labels = read_notes(diagnosis)


# lengths = map(len, labels)
# print(sum(lengths) / len(lengths))
# print(len(icd9_lookup))

## Categories
# cats = []
# with open('notes.csv', 'rb') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         cats.append(row[6])

# cats = list(set(cats))
# example = [None] * len(cats)

# with open('notes.csv', 'rb') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         cat = row[6]
#         idx = cats.index(cat)
#         if row[-1] and random.randint(0, 1) == 1:
#             example[idx] = row[-1]

# for i, cat in enumerate(cats):
#     with open('examples/{}'.format(cat.replace('/', '_')), 'w') as f:
#         f.write(example[i])

##
for text in texts:
    if not text:
        print('oi')
