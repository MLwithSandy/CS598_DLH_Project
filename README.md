# CS598_DLH_Project

---

## Citation to the original paper

```eval
Vitor Pereira, Sérgio Matos, and José Luís Oliveira. 2018. Automated ICD-9-CM medical coding of diabetic patient’s clinical reports. In International Conference on Data Science, E-learning and Information Systems 2018 (DATA’18), October 1–2, 2018, Madrid, Spain. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3279996.3280019
```

---

## Link to the original paper’s repo (if applicable)

```eval
Original code repo is not available. Vitor Pereira (One of the author of original paper) shared the codes via iClould for which url cannot be shared.

Vitor also shared his thesis document, which contained additional information and his research related to the original paper. Link
to the thesis: https://ria.ua.pt/handle/10773/26001

```

---

## Specification of dependencies

We have created python notebooks for each process. You can directly start running the notebook.
List of dependecies:

```eval
google.colab
os
gzip
pickle
gc
numpy
pandas
sklearn.preprocessing
sklearn.metrics
tensorflow.keras.models
string
Levenshtein
collections
gensim.models.word2vec
torch
torch.nn.utils.rnn
tensorflow.keras.preprocessing.sequence
random
```

---

## Data download instruction

```eval
MIMIC III dataset were downloaded from [physionet](https://physionet.org). There is prerequisite course and certification - 'CITI Data or Specimens Only Research' to access MIMIC III dataset via physionet, which should be fulfilled before accessing the MIMIC III dataset.

Direct link to MIMIC III dataset: https://physionet.org/content/mimiciii/1.4/

```

---

## Training code

To run the preprocessing modules, run following steps sequentially:

```eval
1. Open the python notebook: diagonsis_icd_preprocess.ipynb and run all cells for preprocessing of ICD-9 Diagnosis code.

  Data folder: All the data is read and written to data folder, set using variable 'DATA_DIR'.

  Input files:

  - "DIAGNOSES_ICD.csv.gz": MIMIC III dataset  with list of ICD-9 codes

  Output files:
  - "diag_icd9_unique_list.p": List of unique regular ICD-9-CM codes
  - "diag_icd9_rolled_unique_list.p": List of unique rolled up ICD-9-CM codes
  - "diag_icd9.csv.gz": processed regular ICD-9 codes together with k-hot representation
  - "diag_icd9_rolled.csv.gz":  processed rolled ICD-9 codes together with k-hot representation
  - "diag_diabetes_hadm_ids.p": list of admission ids for which the patiant has been diagnosed with diabetes

2. Open the python notebook: notes_preproces.ipynb and run all cells for preprocessing of notes text.

  Data folder: All the data is read and written to data folder, set using variable 'DATA_DIR'.

  Input files:

  - "NOTEEVENTS.csv.gz": MIMIC III dataset with notes text
  - "diag_diabetes_hadm_ids.p": list of admission ids for which the patiant has been diagnosed with diabetes

  Output files:

  - "wordCorpus.p": List of words and their frequencies
  - "notes_tokens_list.p": List of words with frequency more than 5 -> frequent words list
  - "word_token_map.p": File containing mapping of infrequent words to frequent words with least Levestein distance
  - "notes_final.gz": preprocessed and tokenized notes data

3. Open the python notebook: word_embedding.ipynb and run all cells for generating word embeddings and preparing dataset for training and testing of models.

  Data folder: All the data is read and written to data folder, set using variable 'DATA_DIR'.

  Input files:

  - "diag_icd9_unique_list.p": List of unique regular ICD-9-CM codes
  - "diag_icd9_rolled_unique_list.p": List of unique rolled up ICD-9-CM codes
  - "diag_icd9.csv.gz": processed regular ICD-9 codes together with k-hot representation
  - "diag_icd9_rolled.csv.gz":  processed rolled ICD-9 codes together with k-hot representation
  - "diag_diabetes_hadm_ids.p": list of admission ids for which the patiant has been diagnosed with diabetes
  - "notes_tokens_list.p": List of words with frequency more than 5 -> frequent words list
  - "notes_final.gz": preprocessed and tokenized notes data

  Output files:

  - "word2vec_model.model": Word2Vec model
  - "word2vec.vec": Vectors from Word2Vec model
  - "row_index_dictionary.p": word index from trained Word2Vec model
  - "embedding_matrix.p": embedding matrix from trained Word2Vec model
  - "data.npz": Final dataset for training and testing of models, containing data and labels
```

To run the trained models, run:

```eval
1. Open the python notebook: RolledDataModel.ipynb and run all cells.
This file contains 3 models for training: BagOfTricks, CNNBaseline and CNN LayerArchitecture
where the outputs are Rolled ICM codes.
Please make sure the dataset is loaded from correct paths.
```

```eval
2. Open the python notebook: RegularDataModel.ipynb and run all cells.
This file contains 3 models for training: BagOfTricks, CNNBaseline and CNN LayerArchitecture
where the outputs are Regular ICM codes.
Please make sure the dataset is loaded from correct paths.
```

```eval
3. Open the python notebook: CustomRolledDataModel.ipynb and run all cells.
This file contains 3 models for training: BagOfTricks, CNNBaseline and CNN LayerArchitecture
where the outputs are Rolled ICM codes.
These models are custom models trained on top of original paper models.
Please make sure the dataset is loaded from correct paths.
```

```eval
4. Open the python notebook: CustomRegularDataModel.ipynb and run all cells.
This file contains 3 models for training: BagOfTricks, CNNBaseline and CNN LayerArchitecture
where the outputs are Regular ICM codes.
These models are custom models trained on top of original paper models.
Please make sure the dataset is loaded from correct paths.
```

---

## Evaluation code

To evaluate the trained models, run:

```eval
1. Open the python notebook: evaluation.ipynb and run all cells.
Please make sure the pretrained models are loaded from correct paths.

  Data folder: All the data is read and written to data folder, set using variable 'DATA_PATH'.

  Models folder: Trained models from original paper are stored in folder, set with variable 'MODEL_ORIG_PATH' and trained models for ablation is stored in folder, set with variable 'MODEL_PATH'.

```

---

## Pre-trained models

These are the trained models created as part of this project.

These models are replicated with same architecture and hyperparameters as per the original paper.

1. Bag of tricks(BOT) Rolled ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
2. Bag of tricks(BOT) Regular ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
3. Convolutional Neural Network(CNNBaseline) Rolled ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
4. Convolutional Neural Network(CNNBaseline) Regular ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
5. Convolutional Neural Network - 3 Layer Architecture(CNN3Layer) Rolled ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
6. Convolutional Neural Network - 3 Layer Architecture(CNN3Layer) Regular ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)

The below models are custom modifications (trainable initial weights) made on top of the architectures provided in the original paper

1. Bag of tricks(BOT) Rolled ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
2. Bag of tricks(BOT) Regular ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
3. Convolutional Neural Network(CNNBaseline) Rolled ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
4. Convolutional Neural Network(CNNBaseline) Regular ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
5. Convolutional Neural Network - 3 Layer Architecture(CNN3Layer) Rolled ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)
6. Convolutional Neural Network - 3 Layer Architecture(CNN3Layer) Regular ICM code outputs: [Link](https://drive.google.com/file/d/15Bcu1d5Ic5dgPRn7yg8Bh-2oen6gh79B/view?usp=sharing)

## Results

---

### Results with same parameters as per the original paper

| Preprocessing                  | Reprod. | Original |
| ------------------------------ | ------- | -------- |
| Num. of used records           | 399623  | 399623   |
| Num. of regular labels         | 4103    | 4006     |
| Num. of rolled up labels       | 781     | 779      |
| Num. of unique tokens          | 53229   | 53229    |
| Avg. num. of tokens per report | 309.06  | 309.06   |

### Listed below is the comparison of Precision, Recall and F1 Score from original paper and our implementation(reproduction) of BoT and CNN Baseline model.

| Model name                                   | Reprod.             | Original            |
| -------------------------------------------- | ------------------- | ------------------- |
| BoT BaseLine Regular (Precision, Recall, F1) | 63.79, 5.04, 9.34   | 66.25, 8.61, 15.24  |
| CNN Baseline Regular (Precision, Recall, F1) | 72.03, 15.82, 25.94 | 73.97, 25.88, 38.13 |

### Listed below is the comparison of Precision, Recall and F1 Score from original paper and our implementation(reproduction) of CNN and CNN 3-Conv1D Baseline model.

| Model name                                   | Reprod.             | Original            |
| -------------------------------------------- | ------------------- | ------------------- |
| CNN Baseline Regular (Precision, Recall, F1) | 72.03, 15.82, 25.94 | 73.97, 25.88, 38.13 |
| CNN 3-Conv1D Regular (Precision, Recall, F1) | 75.82, 30.71, 43.71 | 76.07, 31.46, 44.51 |

### Listed below is the comparison of our implementation (reproduction) and original paper of ICM Regular codes as well as the ICm rolled up codes for all 6 models:

| Model name                                   | Reprod.             | Original            |
| -------------------------------------------- | ------------------- | ------------------- |
| BoT Baseline Regular (Precision, Recall, F1) | 63.79, 5.04, 9.34   | 66.25, 8.61, 15.24  |
| BoT Baseline Rolled (Precision, Recall, F1)  | 85.67, 10.32, 18.43 | 75.91, 18.89, 30.25 |
| CNN Baseline Regular (Precision, Recall, F1) | 72.03, 15.82, 25.94 | 73.97, 25.88, 38.13 |
| CNN Baseline Rolled (Precision, Recall, F1)  | 77.51, 28.68, 41.87 | 77.73, 35.13, 48.38 |
| CNN 3-Conv1D Regular (Precision, Recall, F1) | 75.82, 30.71, 43.71 | 76.07, 31.46, 44.51 |
| CNN 3-Conv1D Rolled (Precision, Recall, F1)  | 79.56, 38.33, 51.73 | 79.82, 38.26, 51.73 |

### Additional results on top of the original paper(More details in the paper)

| Model name                                   | Replicated Results  | Customized Model results |
| -------------------------------------------- | ------------------- | ------------------------ |
| CNN Baseline Regular (Precision, Recall, F1) | 72.03, 15.82, 25.94 | 74.98, 21.91, 33.91      |
| CNN Baseline Rolled (Precision, Recall, F1)  | 77.51, 28.68, 41.87 | 79.14, 44.66, 57.1       |
| CNN 3-Conv1D Regular (Precision, Recall, F1) | 75.82, 30.71, 43.71 | 75.95, 32.44, 45.46      |
| CNN 3-Conv1D Rolled (Precision, Recall, F1)  | 79.56, 38.33, 51.73 | 81.13, 43.56, 56.68      |
