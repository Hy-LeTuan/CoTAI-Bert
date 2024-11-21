# The CoTAI Bert repository

This repository is a product of my internship at [CoTAI](https://cot.ai/) after researching on Natural Language Processing. Seeing the lack of good tokenizers and Encoder models relating to general and online Vietnamese conversations, in this repository, we aim to introduce **3 engineering feats**:

1. A new dataset that combined many sources of both Vietnamese and English text to train and evaluate an Encoder model
2. A new tokenier adapted to Vietnamese online vocabularies
3. A new Bert model using SOTA optimization techniques to reduce parameters count and increase training efficiency

_This repository also includes training code for the model and processing code for the dataset, so it could be a bit messy_

## Data sources

Below is a data table for all the datasets presented in this repository that are used to train the CoTAI Bert model and the CoTAI tokenizer with a vocabulary size of 50k. All datasets have been downloaded and preprocessed with spam removal and deduplication. To train the model, they are then tokenized with our tokenizer.

| Name                    | Link                                                                              | Size                      | Lang         | Origin              | Note  |
| ----------------------- | --------------------------------------------------------------------------------- | ------------------------- | ------------ | ------------------- | ----- |
| CulturaX                | [Link](https://huggingface.co/datasets/uonlp/CulturaX)                            | $\approx$ 144 GB + 144 GB | Vi + En      | mC4 + OSCAR         | Clean |
| OpenWebText             | [Link](https://openwebtext2.readthedocs.io/en/latest/)                            | 38 GB                     | Multilingual | Reddit posts + Link |       |
| 10000 Vietnamese Books  | [Link](https://www.kaggle.com/datasets/iambestfeeder/10000-vietnamese-books/data) | 1.7 GB                    | Vi           | Books               | Clean |
| Facebook comment Corpus | [Link](https://drive.google.com/file/d/1BNkrAEcUvVO77UJmo82gFM_xySchKG4v/view)    | 3.9 GB                    | Vi           | Facebook groups     | Dirty |
| VOZ                     | [Link](https://huggingface.co/datasets/tarudesu/VOZ-HSD)                          | 1.89 GB                   | Vi           | VOZ forum           | Clean |
| Opus open subtitle      | [Link](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitle)            |                           |              |                     | Dirty |

## Tokenizer

The tokenizer is trained with BPE algorithm, and a vocabulary size of 50k. The data flow in the tokenizer is as follows:

1. A NFC normalizer for Unicode normalization
2. A Regex pattern to split words into chunks, maintaining semantic information when merging. This string is similar to that of the GPT-4 model
3. A BPE model that handles the BPE algorithm
4. A BPE decoder to convert bytes back into words

The tokenizer also includes 5 special tokens, which are added to the end of the vocabulary. These are:

1. CLS token
2. PAD token
3. MASK token
4. SEP token
5. UNK token
