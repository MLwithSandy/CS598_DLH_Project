from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import string


def main():
    full_text = ''

    filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
    tt = string.maketrans(string.digits + filters, 'd' * len(string.digits) + ' ' * len(filters))

    with open('notes.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            text = row[-1].lower().translate(tt).strip()
            full_text += ' ' + text

    with open('full_text.txt', 'w') as f:
        f.write(full_text)


if __name__ == '__main__':
    main()
