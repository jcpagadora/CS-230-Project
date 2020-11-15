from io import open
import torch
from torch.utils.data import TensorDataset
from transformers import RobertaTokenizer

import config

class RawTextData:
    
    def __init__(self, text0, text1, label):
        self.text0 = text0
        self.text1 = text1
        self.label = label


class TokenizedData:
    
    def __init__(self, input_ids, input_mask, segment_ids, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


def get_raw_text(dir_filename, encoding='utf-8'):
    with open(dir_filename, "r", encoding='utf-8') as f:
        raw_text = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            raw_text.append(RawTextData(text0=line[3], text1=line[4], label=line[0]))
    return raw_text


def tokenize_raw_text(text_data, tokenizer, config.max_seq_length):
    pad_token=0
    pad_token_segment_id=0
    features= []
    print("*** Tokenizing Data *** ")
    for i, example in enumerate(text_data):
        if i % 1600 == 1599:
            print("=", end="")
        if i % 64000 == 63999:
            print("[Processed " + str(i+1) + " / " + str(len(text_data)) + "] " )    
        tokens0 = tokenizer.tokenize(example.text0)[:max_seq_length]
        tokens1 = tokenizer.tokenize(example.text1)
        # Truncates the sequence so that its length is at most max_seq_length - 3
        # This takes into account the [SEP] and [CLS] tokens required by BERT
        while len(tokens0) + len(tokens1) > max_seq_length - 3:
            if len(tokens0) > len(tokens1):
                tokens0.pop()
            else:
                tokens1.pop()
        tokens = tokens0 + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        tokens += tokens1 + ["[SEP]"]
        segment_ids += [1] * (len(tokens1) + 1)
        tokens = ["[CLS]"] + tokens
        segment_ids = [1] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # If necessary, zero-pad 
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(TokenizedData(input_ids, input_mask, segment_ids, int(example.label)))
        
    return features


def create_dataset(features):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(input_ids, input_masks, segment_ids, labels)
    return dataset


def get_test_lines(file_dir_name):
    with open(file_dir_name, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            lines.append(line)
    return lines


def lines_to_textdata(lines):
    raw_texts_data = []
    for line in lines:
        raw_texts_data.append(RawTextData(text0=line[3], text1=line[4], label=0))
    return raw_texts_data



