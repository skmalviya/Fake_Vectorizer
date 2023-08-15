import pickle
import numpy as np

file1 = open("Out_embeddings/bert-base-uncasedSentenceTransformer_dev_emb.pkl",'rb')
object_file1 = pickle.load(file1)

list1 = [e for e in object_file1]

file2 = open("Out_embeddings/bert-base-uncasedTransformer_dev_emb.pkl",'rb')
object_file2 = pickle.load(file2)

list2 = [e for e in object_file2]

np.dot(list1[0][1],list2[0][1])/( np.linalg.norm(list1[0][1]) * np.linalg.norm(list2[0][1]))

s = 'shri'