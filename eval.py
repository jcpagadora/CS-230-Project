import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from more_itertools import chunked
from utils import *
import config as cf

def evaluate(model, tokenizer, dataset, lines, output_test_file, batch_size=32):
    """
    Evaluates the model based on classsification accuracy. Receives the logits that are output
    from the network and saves the result in the given output directory file.
    """
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

    print("*** Evaluating ***")
    eval_loss = 0.0
    num_steps = 0
    preds = None
    out_label_ids = None
    for i, batch in enumerate(dataloader):
        if i % 200 == 199:
            print("=", end="")
        if i % 5000 == 4999:
            print("[Step " + str(i+1) + " / " + str(len(dataloader)) + "] " )
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            labels = batch[3]
            outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=labels)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
            
        num_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            
    eval_loss = eval_loss / num_steps
    
    preds_label = np.argmax(preds, axis=1)
    
    accuracy = (preds_label == out_label_ids).mean()
    output_dir = os.path.dirname(output_test_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_test_file, "w") as writer:
        all_logits = preds.tolist()
        for i, logit in enumerate(all_logits):
            line = '<CODESPLIT>'.join(
                [item.encode('ascii', 'ignore').decode('ascii') for item in lines[i]])

            writer.write(line + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
        print("Accuracy =", str(accuracy))

    return accuracy

# ==================================================================================================================

def main():
	try:
		filedir_to_eval = sys.argv[1]
	except IndexError:
		print("Must specify a test file to evaluate on.")
		return
	try:
		output_result_dir = sys.argv[2]
	except IndexError:
		print("Must specify an output file to write results to.")
		return

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	config = RobertaConfig.from_pretrained(cf.model_base, num_labels=cf.num_labels, finetuning_task=cf.finetuning_task)
	tokenizer = RobertaTokenizer.from_pretrained(cf.model_base, do_lower_case=True)
	model = RobertaForSequenceClassification.from_pretrained(cf.model_base, config=config)
	model.to(device)

	model.load_state_dict(torch.load(cf.model_file_dir))

	test_lines = get_test_lines(filefir_to_eval)
	test_rawtexts = lines_to_textdata(test_lines)
	test_features = tokenize_raw_text(test_rawtexts, tokenizer)
	test_dataset = create_dataset(test_features)
	results = evaluate(model, tokenizer, test_dataset, test_lines, output_result_dir, batch_size=cf.eval_batch_size)

	ranks = []
	with open(output_result_dir, encoding='utf-8') as f:
	    batched_data = chunked(f.readlines(), cf.mrr_batch_size)
	    for batch_idx, batch_data in enumerate(batched_data):
	        step1 = batch_data[batch_idx].strip().split('<CODESPLIT>')
	        correct_score = float(step1[-1])
	        scores = np.array([float(data.strip().split('<CODESPLIT>')[-1]) for data in batch_data])
	        rank = np.sum(scores >= correct_score)
	        ranks.append(rank)
	mean_mrr = np.mean(1.0 / np.array(ranks))
	print(" *** Python mrr: %.5f" % (mean_mrr))


if __name__ == "__main__":
	main()


























