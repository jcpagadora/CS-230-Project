import sys
import torch
import json
import random
from pathlib import Path
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel,
                          RobertaTokenizer, RobertaForSequenceClassification)
from utils import *
import config as cf


def train(train_dataset, model, optimizer, batch_size=100, num_epochs=8):
    """ Train the model """
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    t_total = len(train_dataloader) // num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 0, t_total)
    
    print("*** Training ***")

    global_step = 0
    model.zero_grad()
    model.train()
    for idx, _ in enumerate(range(num_epochs)):
        cumu_loss, curr_loss = 0.0, 0.0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                     'labels': batch[3]}
            outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[3])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            loss = outputs[0]
            loss.backward()
            cumu_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            if step % 50 == 49:
                print("=", end="")
            if step % 800 == 799:
                print('[Epoch %d / %d, minibatch %d / %d] loss: %.5f' %
                  (idx + 1, num_epochs, step + 1, len(train_dataloader), (cumu_loss - curr_loss) / 800))
                curr_loss = cumu_loss
    return global_step, cumu_loss / global_step

# ==================================================================================================================

def main():

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	config = RobertaConfig.from_pretrained(cf.model_base, num_labels=cf.num_labels, finetuning_task=cf.finetuning_task)
	tokenizer = RobertaTokenizer.from_pretrained(cf.model_base, do_lower_case=True)
	model = RobertaForSequenceClassification.from_pretrained(cf.model_base, config=config)
	model.to(device)

	train_raw_text = get_raw_text(cf.train_file_dir)

	train_features = tokenize_raw_text(train_raw_text, tokenizer)

	train_dataset = create_dataset(train_features)

	optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)

	global_step, training_loss = train(dataset, model, optimizer)

	torch.save(model.state_dict(), cf.model_file_dir)


if __name__ == "__main__":
	main()

