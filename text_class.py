# train.py

from src.agnews.utils import *
from src.agnews.model import *
from src.agnews.config import Config
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__ == '__main__':

    # python train.py [experiment_name] [noise_std]
    '''   
    The sample configuration using tensorflow privacy is:
    python compute_dp_sgd_privacy.py --N=96000 --batch_size=128 --noise_multiplier=0.621 --epochs=50 --delta=1e-5

    You have to run the code with:
    python text_class.py [experiment_name] [noise_std] [clip]
    '''
    train_file = 'data/ag_news/ag_news.train'
    test_file = 'data/ag_news/ag_news.test'

    print(sys.argv)
    experiment_name = 'non-dp'
    isDP = False

    config = Config()
    if len(sys.argv) < 2:
        print("BiLSTM model")
        experiment_name = 'non-dp'
    elif len(sys.argv) > 2:
        if sys.argv[1] == 'dp':
            print("BiLSTM DP model")
            experiment_name = 'dp'
            noise_multiplier = float(sys.argv[2])
            clip = float(sys.argv[3])
            isDP = True
        elif sys.argv[1] == 'sidp':
            print("BiLSTM SIDP model")
            experiment_name = 'sidp'
            noise_multiplier = float(sys.argv[2])
            clip = float(sys.argv[3])
            isDP = True
            noise_std = float(sys.argv[2])
    else:
        print("Please run the code: python train.py [experiment_name] [noise_std] e.g")
        print("python main_text_classification.py sidp 1.082 7")

    dataset = Dataset(config)
    dataset.load_data(train_file, test_file)

    # Create Model with specified optimizer and loss function
    ##############################################################
    if experiment_name in ['sidp']:
        model = TextSIRNN(config, len(dataset.vocab), dataset.word_embeddings, noise_std)
    else:
        model = TextRNN(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()

    if isDP:
        optimizer = optim.SGD(model.parameters(), lr=config.lr * 100)
        # optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.NLLLoss(reduction='none')

    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.NLLLoss()

    model.add_optimizer(optimizer)
    model.add_loss_op(criterion)
    ##############################################################

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        if experiment_name in ['dp']:
            noise_std = clip * noise_multiplier
            train_loss, val_accuracy = model.run_epoch_dp(dataset.train_iterator, dataset.val_iterator, i, clip,
                                                          noise_std)
        elif experiment_name in ['sidp']:
            sgd_lr = config.lr * 100
            train_loss, val_accuracy = model.run_epoch_sidp(dataset.train_iterator, dataset.val_iterator, i, clip,
                                                            noise_std, sgd_lr, config.batch_size)
            # train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        else:
            train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))
