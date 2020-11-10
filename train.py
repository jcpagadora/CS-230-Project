import torch
import torch.nn as nn
from torch import optim

from load_and_save_data import read_training_files, load_text_processor, load_decoder_inputs, load_encoder_inputs
from models.py import Seq2Seq, init_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_DIM = 800
HIDDEN_SIZE = 1024
BATCH_SIZE = 1000
NUM_EPOCHS = 18

encoder_input_data, encoder_seq_len = load_encoder_inputs('seq2seq/py_train_code_vecs_v2.npy')
decoder_input_data, decoder_target_data = load_decoder_inputs('seq2seq/py_train_docstring_vecs_v2.npy')
num_encoder_tokens, enc_pp = load_text_processor('seq2seq/py_code_processor_v2.dpkl')
num_decoder_tokens, dec_pp = load_text_processor('seq2seq/py_docstring_processor_v2.dpkl')

model = Seq2Seq(Encoder(num_encoder_tokens, emb_dim=EMB_DIM, hidden_size=HIDDEN_SIZE),
                Decoder(num_decoder_tokens, emb_dim=EMB_DIM, hidden_size=HIDDEN_SIZE)).to(device)

model.apply(init_weights)

encoder_input_data = torch.LongTensor(encoder_input_data).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train_code_loader = torch.utils.data.DataLoader(encoder_input_data, batch_size=1000, shuffle=True)
code_data_iter = iter(train_code_loader)

model.train()

for epoch in range(18):

    running_loss = 0.0
    for i, code in enumerate(train_code_loader, 0):
        dec_input = decoder_input_data[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
        dec_target = decoder_target_data[i * BATCH_SIZE : (i+1) * BATCH_SIZE]

        optimizer.zero_grad()

        code = code.type(torch.LongTensor).to(device)
        dec_input = torch.LongTensor(dec_input).to(device)
        dec_target = torch.LongTensor(dec_target).to(device)
        
        code = torch.transpose(code, 0, 1)
        dec_input = torch.transpose(dec_input, 0, 1)
        dec_target = torch.transpose(dec_target, 0, 1)
        outputs = model(code, dec_input)
        outputs = outputs.view(-1, outputs.shape[2])
        dec_target = dec_target.reshape(-1)
        loss = criterion(outputs, dec_target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print("=", end='')
        if i % 300 == 299:    # print every 300 mini-batches
            print('[Epoch %d, minibatch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 300))
            running_loss = 0.0

print('Finished Training')

torch.save(model.state_dict(), 'model.pt')


