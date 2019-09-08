# Pytorch-Torchtext
This is my repository to study TorchText; therefore you might find lots of comment especially on the torchtext part.

Toxic comment classification task, with simple BiLSTM as the model (small data taken from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

Typical process of a task in NLP:
1) Preprocessing and tokenization
2) Vectorization: generating vocabulary of unique tokens and converting words to indices
3) Loading pretrained vectors e.g. Glove, Word2vec, Fasttext
4) Padding text: most of the time the text sequence differs in length
5) Dataloading and batching
6) Model creation and training

Torchtext handles step 1-5 above with minimal code. 

P.S. Train, eval, and test are in main.py

