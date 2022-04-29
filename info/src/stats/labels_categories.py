from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def main():
    unique_cats = ['Discharge Summary', 'Echo', 'ECG', 'Case Management',
                   'Respiratory', 'Nursing', 'Physician', 'Nutrition', 'Pharmacy',
                   'General', 'Social Work', 'Rehab Services', 'Consult', 'Radiology',
                   'Nursing/Other']

    with np.load('icd9_lookup.npz') as l:
        lookup = l['rolled_icd9_lookup']

    with np.load('data.npz') as data:
        x = data['x']
        cats = data['cats']
        y = data['rol_y']

    cats = np.argmax(cats, axis=1)

    # Global
    sums = np.sum(y, axis=0)
    print('Top 10 Indexes:', np.argsort(-sums)[:11])
    print('Top 10 ICD9:', lookup[np.argsort(-sums)[:11]])
    top = set(lookup[np.argsort(-sums)[:10]])

    return

    # Per Category
    sets = []
    for i in xrange(15):
        count = np.sum(y[cats == i], axis=0)
        top11 = lookup[np.argsort(-count)[:11]]
        sets.append(set(top11))
        print(unique_cats[i], ' &', ' & '.join(top11), ' \\\\')

    print('GLOBAL')
    for i, s in enumerate(sets):
        print(unique_cats[i], len(top.difference(s)))

    print('NURSING')
    nursing = sets[5]
    diffs = []
    for i, s in enumerate(sets):
        if i == 5: continue
        print('Nursing -', unique_cats[i], len(nursing.difference(s)))

    print('CASE MANAGEMENT')
    case_mgmt = sets[3]
    diffs = []
    for i, s in enumerate(sets):
        if i == 3: continue
        print('Case Management -', unique_cats[i], len(nursing.difference(s)))

if __name__ == '__main__':
    main()
