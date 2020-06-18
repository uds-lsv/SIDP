# config.py
class Config(object):
    embed_size = 100
    hidden_layers = 1
    hidden_size = 32
    bidirectional = True
    output_size = 4
    max_epochs = 12
    lr = 0.001#0.25
    batch_size = 300
    max_sen_len = 20 # Sequence length for RNN
    dropout_keep = 0.8