import argparse
import pickle
import numpy as np
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help="Pretrained model: [llama-7b_Transformer, bert-base-uncased_Transformer, TfidfVectorizer, CountVectorizer]")
    parser.add_argument('--emb_path', type=str, help="Path where embeddings are stored: 'Out_embeddings/XXX.pkl'")
    args = parser.parse_args()

    print(args)

    # open embedding of train
    fake_claims = open(args.emb_path + args.model + "_train_emb.pkl", 'rb')
    fake_claims = pickle.load(fake_claims)
    # filter - leave only fake claims. We consider that we have database with only fake claims
    fake_claims = [e for e in fake_claims if e[2] == 'REFUTES']
    fake_embs = np.stack([e[1] for e in fake_claims])

    # open test set (do not filter, here we need all types of claims)
    test_claims = open(args.emb_path + args.model + "_dev_emb.pkl", 'rb')
    test_claims = pickle.load(test_claims)
    test_claims = [e for e in test_claims]
    test_embs = np.stack([e[1] for e in test_claims])
    # for each from test
    # calculate distance to nearest claim from train
    # and save it label
    print(f"Total dev data: {len(test_claims)}")
    distances = cosine_similarity(test_embs, fake_embs)
    dist_claims = []
    for i, test_claim in tqdm(enumerate(test_claims)):
        # for test_claim search nearest neighboor from fake_claims and calculate distance
        dist_claims += [{"d": max(distances[i]), "type": test_claim[2]}]
    # save to csv
    pd.DataFrame(dist_claims).to_csv(args.emb_path + args.model + "_distances.csv", index=False)

