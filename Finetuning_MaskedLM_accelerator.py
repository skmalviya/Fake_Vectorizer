# MLM Training With Trainer Link: https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section3_pt.ipynb
# In this script we'll cover the training process for masked-language modeling (MLM) using Accelerator library.
# Accelerate is a library from Hugging Face that simplifies turning PyTorch code for a single GPU into code for
# multiple GPUs, on single or multiple machines.
# Usage:
# python Finetuning_MaskedLM_accelerator.py --data_path data/FEVER/ --model bert-base-uncased --batch_size 64

import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import jsonlines
from datasets import Dataset
import math
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
from tqdm.auto import tqdm

def load_data(json_file):
    claims = []
    labels = []
    with jsonlines.open(json_file) as reader:
        for line in reader:
            claims.append(line['claim'])
            labels.append(line['label'])
    return claims, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="Path of input data: 'data/CORPUS_NAME'")
    parser.add_argument('--model', type=str,
                        help="Pretrained model: [huggyllama/llama-7b, bert-base-uncased, TfidfVectorizer, CountVectorizer]")
    parser.add_argument('--device', default='cuda', type=str, help="Device: 'cpu' or 'cuda'")
    parser.add_argument('--batch_size', default=64, type=int, help="Provide a int number to batchify.")
    args = parser.parse_args()

    print(args)

    train_claim, train_label = load_data(args.data_path + 'train.jsonl')
    dev_claim, dev_label = load_data(args.data_path + 'dev.jsonl')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    print(f"'>>> {args.model} number of parameters: {round(model.num_parameters() / 1_000_000)}M'")

    # Get the Training data
    # with open('clean.txt', 'r') as fp:
    #     text = fp.read().split('\n')

    # Tokenize the text
    tokenized_train = tokenizer(train_claim)
    tokenized_dev = tokenizer(dev_claim, padding=True, truncation=True, return_tensors='pt')

    # A dedicated DataCollatorForLanguageModeling class provided by huggingface just for this task
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Create the DataLoader for traning data
    train_dataloader = DataLoader(
        Dataset.from_dict(tokenized_train),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    # Create the DataLoader for eval data
    eval_dataloader = DataLoader(
        Dataset.from_dict(tokenized_dev), batch_size=args.batch_size, collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Accelerate is a library from Hugging Face that simplifies turning PyTorch code for a single GPU into code for
    # multiple GPUs, on single or multiple machines.
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = 2
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Evaluation before the training
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(args.batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(Dataset.from_dict(tokenized_dev))]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Initial: Perplexity: {perplexity}")

    output_dir = f"Finetuning_MaskedLM_accelerator/{args.model}"

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(64)))

        losses = torch.cat(losses)
        losses = losses[: len(Dataset.from_dict(tokenized_dev))]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

