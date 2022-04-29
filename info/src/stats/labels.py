from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv


def main():
    first_diag = {}
    first_hadm = {}
    hadms = {}

    with open('diagnosis.csv', 'rb') as f:
        reader = csv.reader(f)
        next(f)
        for row in reader:
            patient_id = row[1]
            hadm_id = row[2]
            icd9_code = row[4]

            phadms = hadms.get(patient_id, [])
            if hadm_id not in phadms: phadms.append(hadm_id)
            hadms[patient_id] = phadms

            if patient_id not in first_hadm:
                first_hadm[patient_id] = hadm_id

            if icd9_code[:3] == '250' and patient_id not in first_diag:
                first_diag[patient_id] = hadm_id

    lens = map(len, hadms.itervalues())
    print(sum(lens) / len(lens))
    print(max(lens))

    first = 0
    other = 0
    for (p1, v1), (p2, v2) in zip(first_hadm.iteritems(), first_diag.iteritems()):
        if v1 == v2:
            first += 1
        else:
            other += 1

    print(first)
    print(other)


if __name__ == '__main__':
    main()
