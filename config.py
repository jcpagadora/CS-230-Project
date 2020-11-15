
# Maximum input sequence length for neural neetwork model
max_seq_length=50

# We fine-tuning using a classification task. This is the number of labels for that task.
num_labels = 2

finetuning_task="codesearch"

# Learning rate for the model
learning_rate = 1e-5

# Epsilon hyperparameter for the Adam optimizer
adam_epsilon = 1e-8

# Mini-batch size to use for training/fine-tuning
train_batch_size = 100

# Number of epochs to train for
num_epochs = 8

# Mini-batch size to use for evaluating and saving testing results
eval_batch_size = 32

# Batch size to use when evaluating MRR on test results
mrr_batch_size = 1000

