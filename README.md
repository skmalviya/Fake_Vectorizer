# Fake_Vectorizer:
### Repository for Vectorize Fake-News/Claims through various models, e.g. Llama, Bert, tfidf and count vectorizers

## [FEVER dataset](https://fever.ai/dataset/fever.html)

**Env requirements:**

```
huggingfnltk==3.8.1
nltk==3.8.1
sentence-transformers==2.2.2
numpy==1.25.2
tokenizers==0.13.3
torch==2.0.1
transformers==4.31.0
```

**Download data:**
```
mkdir -p data/FEVER
wget https://fever.ai/download/fever/train.jsonl -O data/FEVER/train.jsonl
wget https://fever.ai/download/fever/shared_task_dev.jsonl -O data/FEVER/dev.jsonl
```

**Run Vectorizer:** It saves the embeddings in pickle file inside the *Out_embeddings* directory:
```
python Text_vectorizer_Transformer.py --data_path data/FEVER/ --model bert-base-uncased \
--output_path Out_embeddings/
```

**Find Distance:** It finds the most similar claim in training data for a given claim in dev data:
```
python Fake_distance.py --model bert-base-uncased --emb_path Out_embeddings/
```

**FineTuning:** It finetunes a transformer model, e.g. bert-base-uncased, roberta_base, on the training data. The finetuned model will later be used to generate the embedding and distances:
  1. Following [link](https://github.com/jamescalam/transformers/blob/main/course/training/04_mlm_training_Trainer.ipynb), masked-language modeling (MLM) using the HuggingFace `Trainer` function:

  ```
  python FineTuning_MaskedLM.py --data_path data/FEVER/ --model bert-base-uncased --batch_size 64
  ```

  2. Following [link](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section3_pt.ipynb), masked-language modeling (MLM) using the HuggingFace `Trainer` and `DataCollatorForLanguageModeling` functions:

  ```
  python Finetuning_MaskedLM_data_collator.py --data_path data/FEVER/ --model bert-base-uncased \
   --batch_size 64
  ```
  3. Following [link](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section3_pt.ipynb), masked-language modeling (MLM) using `Accelerator` library:

  ```
  python Finetuning_MaskedLM_accelerator.py --data_path data/FEVER/ --model bert-base-uncased \
   --batch_size 64
  ```
