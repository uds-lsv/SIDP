# model.py

import torch
from torch import nn
import numpy as np
from src.agnews.utils import *
from src.dp_layers.dp_layers import SILinear, SILSTM
from numpy import linalg as LA

class TextSIRNN(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings, noise_std=0.0):
        super(TextSIRNN, self).__init__()
        self.config = config
        self.noise_std = noise_std
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.lstm = SILSTM(input_size=self.config.embed_size,
                                       hidden_size=self.config.hidden_size,
                                       noise_std=self.noise_std,
                                       num_layers=self.config.hidden_layers,
                                       bidirectional=self.config.bidirectional)

        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc = SILinear(
            self.config.hidden_size * self.config.hidden_layers * (1 + self.config.bidirectional),
            self.config.output_size, bias=True, noise_std=self.noise_std
        )

        self.lay_norm = nn.LayerNorm(self.config.output_size, elementwise_affine=False)

        # Softmax non-linearity
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x, noise_std=0.0):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)
        self.noise_std = noise_std
        if self.noise_std > 0:
            # print("I reach here")
            lstm_out, (h_n, c_n) = self.lstm(embedded_sent, noise_std=self.noise_std)
        else:
            lstm_out, (h_n, c_n) = self.lstm(embedded_sent)
        final_feature_map = self.dropout(h_n)  # shape=(num_layers * num_directions, 64, hidden_size)

        # Convert input to (64, hidden_size * hidden_layers * num_directions) for linear layer
        final_feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1)
        final_out = self.fc(final_feature_map)
        final_out = self.lay_norm(final_out)

        return self.logsoftmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

        return train_losses, val_accuracies

    def run_epoch_sidp(self, train_iterator, val_iterator, epoch, clip, noise_multiplier, lr, batch_size):
        # sidp
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        sgd_lr = lr
        for g in self.optimizer.param_groups:
            sgd_lr = g['lr']

        dp_std = sgd_lr * clip * noise_multiplier / batch_size

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x, dp_std)
            loss = self.loss_op(y_pred, y)

            saved_var = dict()
            for tensor_name, tensor in self.named_parameters():
                if tensor_name == "embeddings.weight": continue
                saved_var[tensor_name] = torch.zeros_like(tensor)

            # print(saved_var.keys())

            for j in loss:
                j.backward(retain_graph=True)
                # param_norm_before = compute_norm(model, norm_type=2)
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip, norm_type=2)
                # param_norm_after = compute_norm(model, norm_type=2)
                # total_diff += np.abs(param_norm_after - param_norm_before)
                # print(self.named_parameters())
                for tensor_name, tensor in self.named_parameters():
                    if tensor_name == "embeddings.weight": continue
                    new_grad = tensor.grad
                    saved_var[tensor_name].add_(new_grad)
                self.optimizer.zero_grad()

            for tensor_name, tensor in self.named_parameters():
                if tensor_name == "embeddings.weight": continue
                # saved_var[tensor_name].add_(tensor.grad)
                # noise = std_grad * torch.randn_like(tensor.grad)
                # saved_var[tensor_name].add_(noise)
                tensor.grad = saved_var[tensor_name] / loss.shape[0]

            # loss.backward()
            losses.append(loss.data.mean().cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

        return train_losses, val_accuracies


class TextRNN(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(TextRNN, self).__init__()
        self.config = config

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.lstm = nn.LSTM(input_size=self.config.embed_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.hidden_layers,
                            # dropout = self.config.dropout_keep,
                            bidirectional=self.config.bidirectional)

        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size * self.config.hidden_layers * (1 + self.config.bidirectional),
            self.config.output_size
        )

        # Softmax non-linearity
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)

        lstm_out, (h_n, c_n) = self.lstm(embedded_sent)
        final_feature_map = self.dropout(h_n)  # shape=(num_layers * num_directions, 64, hidden_size)

        # Convert input to (64, hidden_size * hidden_layers * num_directions) for linear layer
        final_feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1)
        final_out = self.fc(final_feature_map)
        return self.logsoftmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

        return train_losses, val_accuracies

    def run_epoch_dp(self, train_iterator, val_iterator, epoch, clip, std_grad):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)

            saved_var = dict()
            for tensor_name, tensor in self.named_parameters():
                if tensor_name == "embeddings.weight": continue
                saved_var[tensor_name] = torch.zeros_like(tensor)

            # print(saved_var.keys())

            for j in loss:
                j.backward(retain_graph=True)
                # param_norm_before = compute_norm(model, norm_type=2)
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip, norm_type=2)
                # param_norm_after = compute_norm(model, norm_type=2)
                # total_diff += np.abs(param_norm_after - param_norm_before)
                # print(self.named_parameters())
                for tensor_name, tensor in self.named_parameters():
                    if tensor_name == "embeddings.weight": continue
                    new_grad = tensor.grad
                    saved_var[tensor_name].add_(new_grad)
                self.optimizer.zero_grad()

            for tensor_name, tensor in self.named_parameters():
                if tensor_name == "embeddings.weight": continue
                # saved_var[tensor_name].add_(tensor.grad)
                noise = std_grad * torch.randn_like(tensor.grad)
                saved_var[tensor_name].add_(noise)
                tensor.grad = saved_var[tensor_name] / loss.shape[0]

            # loss.backward()
            losses.append(loss.data.mean().cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

        return train_losses, val_accuracies