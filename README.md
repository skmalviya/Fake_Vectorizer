# Fake_Vectorizer:
### Repository for Vectorize Fake-News/Claims through various models, e.g. Llama, Bert, tfidf and count vectorisers   

## [FEVER dataset](https://fever.ai/dataset/fever.html)
Download data:
```
mkdir -p data/FEVER
wget https://fever.ai/download/fever/train.jsonl -O data/FEVER/train.jsonl
wget https://fever.ai/download/fever/shared_task_dev.jsonl -O data/FEVER/dev.jsonl
```

Env requirements:

```
huggingfnltk==3.8.1
nltk==3.8.1
sentence-transformers==2.2.2
numpy==1.25.2
tokenizers==0.13.3
torch==2.0.1
transformers==4.31.0
```

Run Vectorizer. It saves the embedding in pickle file inside the *Out_embeddings* directory:
```
python Text_vectorizer_Transformer.py --data_path data/FEVER/ --model bert-base-uncased \
--output_path Out_embeddings/
```
