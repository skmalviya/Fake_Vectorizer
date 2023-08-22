# MLM Training With Trainer Link: https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section3_pt.ipynb
# In this script we'll cover the training process for masked-language modeling (MLM) using the 
# HuggingFace `Trainer` and `DataCollatorForLanguageModeling` functions.
# DataCollatorForLanguageModeling class provided by huggingface just for this task.
# Usage:
# python Finetuning_MaskedLM_data_collator.py --data_path data/FEVER/ --model bert-base-uncased --batch_size 64

import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import jsonlines
from datasets import Dataset
import math


def load_data(json_file):
    claims = []
    labels = []
    with jsonlines.open(json_file) as reader:
        for line in reader:
            claims.append(line['claim'])
            labels.append(line['label'])
    return claims, labels

# We’ll first tokenize our corpus as usual, but without setting the truncation=True option in our
# tokenizer. We’ll also grab the word IDs if they are available ((which they will be if we’re using a fast tokenizer,
# as described in Chapter 6), as we will need them later on to do whole word masking. We’ll wrap this in a simple
# function, and while we’re at it we’ll remove the text and label columns since we don’t need them any longer:
def tokenize_function(tokenizer, examples):
    result = tokenizer(examples)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="Path of input data: 'data/CORPUS_NAME'")
    parser.add_argument('--model', type=str,
                        help="Pretrained model: [huggyllama/llama-7b, bert-base-uncased, TfidfVectorizer, CountVectorizer]")
    parser.add_argument('--device', default='cuda', type=str, help="Device: 'cpu' or 'cuda'")
    parser.add_argument('--batch_size', default=64, type=int, help="Provide a int number to batchify.")
    args = parser.parse_args()

    print(args)

    # Load the data
    train_claim, train_label = load_data(args.data_path + 'train.jsonl')
    dev_claim, dev_label = load_data(args.data_path + 'dev.jsonl')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    print(f"'>>> {args.model} number of parameters: {round(model.num_parameters() / 1_000_000)}M'")

    # Get the Training data
    # with open('clean.txt', 'r') as fp:
    #     text = fp.read().split('\n')

    # Tokenize the text
    tokenized_train = tokenize_function(tokenizer, train_claim)
    tokenized_dev = tokenize_function(tokenizer, dev_claim)

    # A dedicated DataCollatorForLanguageModeling class provided by huggingface just for this task
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # We'll pass a training `args` dictionary to the `Trainer` defining our training arguments.
    training_args = TrainingArguments(
        output_dir=f"Finetuning_MaskedLM_data_collator/{args.model}",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=2,
        fp16=True,
        report_to="none"
    )

    # Now we'll import and initialize our `Trainer`.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_dict(tokenized_train),
        eval_dataset=Dataset.from_dict(tokenized_dev),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Initital eval before finetuning the model
    eval_results = trainer.evaluate()
    print(f">>> Initial eval_results: {eval_results}")
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # And train.
    trainer.train()

    # Final eval after finetuning the model
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Save the model
    model.save_pretrained(f"Finetuning_MaskedLM_data_collator/{args.model}/")
    tokenizer.save_pretrained(f"Finetuning_MaskedLM_data_collator/{args.model}/")
