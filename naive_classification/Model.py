import torch
import torch.nn as nn

import numpy as np
from collections import OrderedDict


class EarlyStopping:
    def __init__(self, patience=0, moniter='val_loss', min_delta=0):
        self.moniter = moniter
        self.early_stop = False

        self._patience = patience
        self._min_delta = min_delta

        self._counter = 0
        self._best_score = None

        if 'loss' in self.moniter:
            self._min_delta *= -1
            self._moniter_operation = np.less
        elif 'accuracy' in self.moniter:
            self._min_delta *= 1
            self._moniter_operation = np.greater

    def __call__(self, score):
        if self._best_score == None:
            self._best_score = score

        if self._moniter_operation(score - self._min_delta, self._best_score):
            self._best_score = score
            self._counter = 0
        else:
            self._counter += 1

            if self._counter >= self._patience:
                self.early_stop = True


class Model:
    def __init__(self, net, optimizer, criterion):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = self.net.to(self.device)

    def train(self, train_loader, epochs, val_loader=None, scheduler=None, early_stopping=None):
        history = {'loss': [], 'accuracy': []}
        if val_loader != None:
            history['val_loss'] = []
            history['val_accuracy'] = []

        length = len(str(epochs))

        for epoch in range(epochs):
            self.net.train()

            correct = 0
            loss = 0.0
            for i, (x_train, y_train) in enumerate(train_loader):
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                self.optimizer.zero_grad()

                output = self.net(x_train)
                _, y_hat = output.max(dim=1)
                correct += (y_hat == y_train).sum().item()

                temp_loss = self.criterion(output, y_train)
                loss += temp_loss.item()

                temp_loss.backward()
                self.optimizer.step()

                current_progress = (i + 1) / len(train_loader) * 100
                progress_bar = '=' * int((i + 1) * (20 / len(train_loader)))
                print(f'\rEpochs: {(epoch + 1):>{length}} / {epochs}, [{progress_bar:<20}] {current_progress:>6.2f}%, loss: {temp_loss.item():.3f}', end='')

            loss /= len(train_loader)
            accuracy = correct / len(train_loader.dataset)

            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

            print(f'\rEpochs: {(epoch + 1):>{length}} / {epochs}, [{"=" * 20}], ', end='')

            if val_loader == None:
                print(f'loss: {loss:.3f}, accuracy: {accuracy:.3f}')
            else:
                print(f'loss: {loss:.3f}, accuracy: {accuracy:.3f}, ', end='')
                val_history = self.test(val_loader, is_test=False, verbose=False)

                history['val_loss'].append(val_history['loss'])
                history['val_accuracy'].append(val_history['accuracy'])

                if early_stopping != None:
                    early_stopping(history[early_stopping.moniter][-1])

                    if early_stopping.early_stop == True:
                        print('Sweet Point')
                        break

            if scheduler != None:
                scheduler.step()

        return history

    def test(self, test_loader, is_test=True, verbose=True):
        self.net.eval()

        history = {}

        loss = 0.0
        correct = 0

        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(test_loader):
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)

                output = self.net(x_test)
                _, y_hat = output.max(dim=1)
                correct += (y_hat == y_test).sum().item()

                temp_loss = self.criterion(output, y_test).item()
                loss += temp_loss

                if verbose:
                    current_progress = (i + 1) / len(test_loader) * 100
                    progress_bar = '=' * int((i + 1) * (20 / len(test_loader)))
                    print(f'\rTest: [{progress_bar:<20}] {current_progress:6.2f}%, loss: {temp_loss:.3f}', end='')

        loss /= len(test_loader)
        accuracy = correct / len(test_loader.dataset)

        history['loss'] = loss
        history['accuracy'] = accuracy

        prefix = 'test' if is_test else 'val'
        if verbose:
            print(f'\rTest: [{"=" * 20}], ', end='')
        print(f'{prefix}_loss: {loss:>.3f}, {prefix}_accuracy: {accuracy:.3f}')

        return history

    def predict(self, x):
        self.net.eval()

        x = torch.Tensor(x).to(self.device)

        output = self.net(x)

        return np.array(output.tolist())

    def predict_class(self, x):
        self.net.eval()

        x = torch.Tensor(x).to(self.device)

        output = self.net(x)
        _, y_hat = output.max(dim=1)

        return np.array(y_hat.tolist())

    def summary(self, input_shape):
        hooks = []
        summary_dict = OrderedDict()

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_index = len(summary_dict)

                key = f'{class_name}-{module_index + 1}'
                summary_dict[key] = {}
                summary_dict[key]['input_shape'] = list(input[0].shape)
                summary_dict[key]['output_shape'] = list(output[0].shape)

                train_parameters = 0
                total_parameters = 0
                for parameter in module.parameters():
                    amount = np.prod(parameter.shape)

                    if parameter.requires_grad:
                        train_parameters += amount
                    total_parameters += amount
                summary_dict[key]['train_parameters'] = train_parameters
                summary_dict[key]['total_parameters'] = total_parameters

            if (not isinstance(module, nn.Sequential)) and (not isinstance(module, nn.ModuleList)) and (not (module == self.net)):
                hooks.append(module.register_forward_hook(hook))

        self.net.apply(register_hook)

        input_data = torch.randn(input_shape).to(self.device)
        self.net(input_data)

        for i in hooks:
            i.remove()

        max_key_length = max(map(len, summary_dict.keys())) + 1

        print('-' * (65 + max_key_length))
        print(f'{"Layer":>{max_key_length}}{"Input Shape":>25}{"Output Shape":>25}{"Parameters":>15}')
        print('=' * (65 + max_key_length))

        train_parameters = 0
        total_parameters = 0
        for key, value in summary_dict.items():
            print(f'{key:>{max_key_length}}{str(value["input_shape"]):>25}{str(value["output_shape"]):>25}{value["train_parameters"]:>15}')

            train_parameters += value['train_parameters']
            total_parameters += value['total_parameters']

        print('=' * (65 + max_key_length))
        print(f'Train Parameters: {train_parameters}, Total Parameters: {total_parameters}')
        print('-' * (65 + max_key_length))

    def export(self, filename, input_shape):
        input_data = torch.randn(input_shape).to(self.device)

        torch.onnx.export(
            self.net,
            input_data,
            filename,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
