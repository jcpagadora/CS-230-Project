import os
import urllib.request
from pathlib import Path
from ktext.preprocess import processor
import dill as dpickle
import numpy as np

processed_data_filenames = [
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test.docstring',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test_original_function.json.gz',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train.docstring',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train_original_function.json.gz',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.docstring',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid_original_function.json.gz',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings_original_function.json.gz']

outpath = Path('processed_data')
print(f'Saving files to {str(outpath.absolute())}')
for url in processed_data_filenames:
    print(f'downloading {url}')
    filename = os.path.join(outpath, url[66:])
    urllib.request.urlretrieve(url, filename)

train_code, holdout_code, train_docstring, holdout_docstring = read_training_files('processed_data/')
save_processors(train_code, train_docstring)


def read_training_files(data_path:str):
    # Read data from directory
    PATH = Path(data_path)
    with open(PATH/'train.function', 'r') as f:
        train_code1 = f.readlines()
    with open(PATH/'valid.function', 'r') as f:
        train_code2 = f.readlines()
    train_code = train_code1 + train_code2
    with open(PATH/'test.function', 'r') as f:
        holdout_code = f.readlines()
    with open(PATH/'train.docstring', 'r') as f:
        train_docstring1 = f.readlines()
    with open(PATH/'valid.docstring', 'r') as f:
        train_docstring2 = f.readlines()
    train_docstring = train_docstring1 + train_docstring2
    with open(PATH/'test.docstring', 'r') as f:
        holdout_docstring= f.readlines()
    return train_code, holdout_code, train_docstring, holdout_docstring


def save_processors(code, docstring)
    code_processor = processor(heuristic_pct_padding=.7, keep_n=20000)
    code_pp = code_processor.fit_transform(code)
    docstring_processor = processor(append_indicators=True, heuristic_pct_padding=.7, keep_n=14000, padding ='post')
    docstring_pp = docstring_processor.fit_transform(docstring)
    with open('seq2seq/py_code_processor_v2.dpkl', 'wb') as f:
    	dpickle.dump(code_processor, f)
    with open('seq2seq/py_docstring_processor_v2.dpkl', 'wb') as f:
    	dpickle.dump(docstring_processor, f)
	np.save('seq2seq/py_train_code_vecs_v2.npy', train_code)
	np.save('seq2seq/py_train_docstring_vecs_v2.npy', train_docstring)

def load_text_processor(fname='title_pp.dpkl'):
    with open(fname, 'rb') as f:
        pp = dpickle.load(f)
    num_tokens = max(pp.id2token.keys()) + 1
    return num_tokens, pp

def load_decoder_inputs(decoder_np_vecs='train_title_vecs.npy'):
    vectorized_title = np.load(decoder_np_vecs)
    decoder_input_data = vectorized_title[:, :-1]
    decoder_target_data = vectorized_title[:, 1:]
    return decoder_input_data, decoder_target_data

def load_encoder_inputs(encoder_np_vecs='train_body_vecs.npy'):
    vectorized_body = np.load(encoder_np_vecs)
    encoder_input_data = vectorized_body
    doc_length = encoder_input_data.shape[1]
    return encoder_input_data, doc_length


















    







