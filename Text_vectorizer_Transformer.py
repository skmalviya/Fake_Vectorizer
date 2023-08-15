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
from transformers import AutoModel, LlamaModel, AutoTokenizer, LlamaTokenizer
import torch

## for bag-of-words
# from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


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


def batchify(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="Path of input data: 'data/CORPUS_NAME'")
    parser.add_argument('--model', type=str,
                        help="Pretrained model: [huggyllama/llama-7b, bert-base-uncased, TfidfVectorizer, CountVectorizer]")
    parser.add_argument('--output_path', type=str, help="Path to store output file: 'Out_embeddings/XXX.jsonl'")
    parser.add_argument('--device', default='cuda', type=str, help="Device: 'cpu' or 'cuda'")
    parser.add_argument('--batch_size', default=64, type=int, help="Provide a int number to batchify.")
    args = parser.parse_args()

    print(args)

    train_claim, train_label = load_data(args.data_path + 'train.jsonl')
    dev_claim, dev_label = load_data(args.data_path + 'dev.jsonl')

    vocab = [s.split() for s in train_claim + dev_claim]
    vocab = [w for subl in vocab for w in subl]
    print(f"Total word count: {len(vocab)}")
    vocab = set(vocab)
    print(f"Approx vocab size: {len(vocab)}")

    train_claim = pre_process(train_claim)
    dev_claim = pre_process(dev_claim)

    vocab = [s.split() for s in train_claim + dev_claim]
    vocab = [w for subl in vocab for w in subl]
    print(f"Total word count after preprocessing: {len(vocab)}")
    vocab = set(vocab)
    print(f"Approx vocab size after preprocessing: {len(vocab)}")

    print('\nTraining data (first 5)...\n', train_claim[:5], train_label[:5], '\n')
    print('Dev data (first 5)...\n', dev_claim[:5], dev_label[:5], '\n')

    print('Preprocessing done!!!\n')

    print('Vectorization starts...')
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
        # model = SentenceTransformer(args.model)

        # Load AutoModel from huggingface model repository
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(args.model).to(args.device)
        # if 'llama' in args.model:
        #     print("\nLoading llama tokenizer and model...")
        #     tokenizer = LlamaTokenizer.from_pretrained(args.model)
        #     model = LlamaModel.from_pretrained(args.model)

        # Batchify the data
        train_claim_batch = batchify(train_claim, args.batch_size)
        dev_claim_batch = batchify(dev_claim, args.batch_size)

        # Compute token embeddings
        with torch.no_grad():
            train_emb = []
            for b in train_claim_batch:
                # Tokenize sentences
                train_input = tokenizer(b, padding=True, truncation=True, return_tensors='pt').to(args.device)
                train_emb += mean_pooling(model(**train_input), train_input['attention_mask'])
            train_emb = torch.stack(train_emb).cpu().numpy()

            dev_emb = []
            for b in dev_claim_batch:
                dev_input = tokenizer(b, padding=True, truncation=True, return_tensors='pt').to(args.device)
                dev_emb += mean_pooling(model(**dev_input), dev_input['attention_mask'])
            dev_emb = torch.stack(dev_emb).cpu().numpy()

        # Our claims we like to encode
        # train_emb = model.encode(train_claim, show_progress_bar=True)
        # dev_emb = model.encode(dev_claim, show_progress_bar=True)
    print('Vectorization done!!!\n')

    print('Saving the embeddings...')
    assert (train_emb.shape[0] == len(train_claim) and train_emb.shape[0] == len(train_label))
    assert (dev_emb.shape[0] == len(dev_claim) and dev_emb.shape[0] == len(dev_label))

    # Write embeddings to a jsonl file
    write_out(zip(train_claim, train_emb, train_label), args.output_path + args.model + '_train_emb.pkl')
    write_out(zip(dev_claim, dev_emb, dev_label), args.output_path + args.model + '_dev_emb.pkl')

    print(f'Done Vectorization with {args.model} !!!\n')
