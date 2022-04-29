from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def main():
    unique_cats = ['Disch. Summ.', 'Echo', 'ECG', 'Case Mngmt. ',
                   'Respiratory ', 'Nursing', 'Physician ', 'Nutrition', 'Pharmacy',
                   'General', 'Social Work', 'Rehab Svcs.', 'Consult', 'Radiology',
                   'Nursing/Oth']
    counts = np.zeros((len(unique_cats)))

    with np.load('data.npz') as data:
        cats_train = data['cats_train']
        cats_val = data['cats_val']
        cats_test = data['cats_test']

    cats = np.vstack([cats_train, cats_val, cats_test])
    cats_sum = np.sum(cats, axis=0)
    percents = cats_sum / np.sum(cats_sum) * 100

    print('Nursing:', (cats_sum[5] + cats_sum[-1]) / np.sum(cats_sum))
    print('Exams:', (cats_sum[1] + cats_sum[2] + cats_sum[4] + cats_sum[13]) / np.sum(cats_sum))

    cm = plt.cm.get_cmap('tab20')
    colors = cm(np.arange(len(unique_cats)) / len(unique_cats))
    # colors = cm(np.linspace(0, 1, len(unique_cats)))

    patches, texts = plt.pie(cats_sum, colors=colors, startangle=90)
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(unique_cats, percents)]
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, cats_sum),
                                          key=lambda x: x[2],
                                          reverse=True))
    plt.legend(patches, labels, loc='upper left', bbox_to_anchor=(-.3, 1), fontsize=8)
    plt.axis('equal')
    plt.savefig('chart_report_categories.png', bbox_inches='tight', dpi=900)
    # plt.show()


if __name__ == '__main__':
    main()
