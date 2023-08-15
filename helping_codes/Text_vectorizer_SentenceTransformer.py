"""
This script vectorize a piece of text into an embedding

Usage: python Text_vectorizer.py --data_path data/FEVER/ --model bert-base-uncased --output_path Out_embeddings/ 
"""

from sentence_transformers import SentenceTransformer
import argparse
import jsonlines
import pickle
from pathlib import Path
import tqdm
import string
from nltk.corpus import stopwords
import numpy as np

## for bag-of-words
# from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def load_data(json_file):
    claims = []
    labels = []
    with jsonlines.open(json_file) as reader:
        for line in reader:
            claims.append(line['claim'])
            labels.append(line['label'])
    return claims, labels


def write_out(lines, out_file):
    if not Path(out_file[:out_file.rindex('/')]).exists():
        Path(out_file[:out_file.rindex('/')]).mkdir(parents=True, exist_ok=True)  # Create directories if don't exist!
    if 'pkl' in out_file:
        with open(out_file, 'wb') as pkl_writer:
            pickle.dump(lines, pkl_writer)
    elif 'jsonl' in out_file:
        with jsonlines.open('out_file', mode='w') as writer:
            for l in lines:
                writer.write(l)


def pre_process(text_list):
    def pre_process_(text):
        text = text.lower()

        PUNCT_TO_REMOVE = string.punctuation
        text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

        STOPWORDS = set(stopwords.words('english'))
        text = " ".join([word for word in str(text).split() if word not in STOPWORDS])

        return text

    if isinstance(text_list, list):
        return [pre_process_(t) for t in text_list]
    else:
        return pre_process_(text_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="Path of input data: 'data/CORPUS_NAME'")
    parser.add_argument('--model', type=str,
                        help="Pretrained model: [huggyllama/llama-7b, bert-base-uncased, TfidfVectorizer, CountVectorizer]")
    parser.add_argument('--output_path', type=str, help="Path to store output file: 'Out_embeddings/XXX.jsonl'")
    args = parser.parse_args()

    print(args)

    train_claim, train_label = load_data(args.data_path + 'train.jsonl')
    dev_claim, dev_label = load_data(args.data_path + 'dev.jsonl')

    vocab = [s.split() for s in train_claim + dev_claim]
    vocab = [w for subl in vocab for w in subl]
    print(f"Total word count: {len(vocab)}")
    vocab = set(vocab)
    print(f"Approx vocab size: {len(vocab)}")

    # train_claim = pre_process(train_claim)
    # dev_claim = pre_process(dev_claim)

    vocab = [s.split() for s in train_claim + dev_claim]
    vocab = [w for subl in vocab for w in subl]
    print(f"Total word count after preprocessing: {len(vocab)}")
    vocab = set(vocab)
    print(f"Approx vocab size after preprocessing: {len(vocab)}")

    print('\nTraining data (first 5)...\n', train_claim[:5], train_label[:5], '\n')
    print('Dev data (first 5)...\n', dev_claim[:5], dev_label[:5], '\n')

    if args.model == 'TfidfVectorizer':
        ## Tf-Idf (advanced variant of BoW)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(train_claim)
        train_emb = np.asarray(vectorizer.transform(train_claim).todense())

        dic_vocabulary = vectorizer.vocabulary_
        print(f"Vectorizer Vocabulary size: {len(dic_vocabulary)}")
        print(f"Dev sample: {dev_claim[0]} {vectorizer.transform([dev_claim[0]])}")
        dev_emb = np.asarray(vectorizer.transform(dev_claim).todense())

    elif args.model == 'CountVectorizer':
        ## Count (classic BoW)
        vectorizer = CountVectorizer()
        vectorizer.fit(train_claim)
        train_emb = np.asarray(vectorizer.transform(train_claim).todense())

        dic_vocabulary = vectorizer.vocabulary_
        print(f"Vectorizer Vocabulary size: {len(dic_vocabulary)}")
        print(f"Dev sample: {dev_claim[0]} {vectorizer.transform([dev_claim[0]])}")
        dev_emb = np.asarray(vectorizer.transform(dev_claim).todense())
    else:
        model = SentenceTransformer(args.model)

        # Our claims we like to encode
        train_emb = model.encode(train_claim, show_progress_bar=True)
        dev_emb = model.encode(dev_claim, show_progress_bar=True)

    assert (train_emb.shape[0] == len(train_claim) and train_emb.shape[0] == len(train_label))
    assert (dev_emb.shape[0] == len(dev_claim) and dev_emb.shape[0] == len(dev_label))

    # Write embeddings to a jsonl file
    write_out(zip(train_claim, train_emb, train_label), args.output_path + args.model + '_SentenceTransformer_train_emb.pkl')
    write_out(zip(dev_claim, dev_emb, dev_label), args.output_path + args.model + '_SentenceTransformer_dev_emb.pkl')

    print(f'Done Vectorizatoin with {args.model}...')
