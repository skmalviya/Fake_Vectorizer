# MLM Training With Trainer Link: https://github.com/jamescalam/transformers/blob/main/course/training/04_mlm_training_Trainer.ipynb
# In this script we'll cover the training process for masked-language modeling (MLM) using the HuggingFace `Trainer` function.
# Usage:
# python FineTuning_MaskedLM.py --data_path data/FEVER/ --model bert-base-uncased --batch_size 64

import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
import torch
import jsonlines
import math


def load_data(json_file):
    claims = []
    labels = []
    with jsonlines.open(json_file) as reader:
        for line in reader:
            claims.append(line['claim'])
            labels.append(line['label'])
    return claims, labels


def mask_data(inputs, tokenizer):
    # Now we mask tokens in the *input_ids* tensor, using the 15% probability we used before - and the not a [CLS],
    # [SEP], or [PAD] token condition.

    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != tokenizer.cls_token_id) * \
               (inputs.input_ids != tokenizer.sep_token_id) * (inputs.input_ids != 0)

    # And now we take the indices of each `True` value, within each individual vector.
    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    # Now apply these indices to each respective row in 'input_ids', assigning each of the values at these indices as [103].
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = tokenizer.mask_token_id

    return inputs


# The `Trainer` expects a `Dataset` object, so we need to initialize this.
class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


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

    # Get the Training data
    # with open('clean.txt', 'r') as fp:
    #     text = fp.read().split('\n')

    # Tokenize the text
    train_inputs = tokenizer(train_claim[:1000], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    dev_inputs = tokenizer(dev_claim[:100], return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    # Then we create our *labels* tensor by cloning the *input_ids* tensor.
    train_inputs['labels'] = train_inputs.input_ids.detach().clone()
    dev_inputs['labels'] = dev_inputs.input_ids.detach().clone()

    # Now we mask tokens in the *input_ids* tensor, using the 15% probability we used before - and the not a [CLS],
    # [SEP], or [PAD] token condition.
    train_inputs_masked = mask_data(train_inputs, tokenizer)
    dev_inputs_masked = mask_data(dev_inputs, tokenizer)

    # Initialize our data using the `FakeDataset` class.
    train_dataset = FakeDataset(train_inputs_masked)
    dev_dataset = FakeDataset(dev_inputs_masked)

    # We'll pass a training `args` dictionary to the `Trainer` defining our training arguments.
    train_args = TrainingArguments(
        output_dir=f"FineTuning_MaskedLM/{args.model}/",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=2,
        report_to="none"
    )

    # Now we'll import and initialize our `Trainer`.
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
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
    model.save_pretrained(f"FineTuning_MaskedLM/{args.model}/")
    tokenizer.save_pretrained(f"FineTuning_MaskedLM/{args.model}/")
