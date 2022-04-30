"""
base util functions
parse_setting, train, train_evaluate, test
"""
from pathlib import Path
import sys
import itertools
from typing import Callable
import time

import torch
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms


__all__ = ['parse_setting', 'get_Train_Test_loaders', 'train', 'train_evaluate', 'test', '_animate']


def _animate(done: Callable):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done():
            break
        sys.stdout.write(f'\revaluating... ' + c)
        sys.stdout.flush()  # flush buffer
        time.sleep(0.1)
    sys.stdout.write('\rDone!          \n')


# Training settings
def parse_setting(file_dir: str = None):
    """
    usage: student_mnist.py [-h] [--batch-size N] [--test-batch-size N]
                            [--epochs N] [--lr LR] [--momentum M] [--no-cuda]
                            [--seed S] [--log-interval N]
    """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # extend
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if file_dir is not None:
        print(file_dir, args)

    return args


def get_Train_Test_loaders(where_to_download: Path, **kwargs):
    mnist_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # MNIS 1d
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(where_to_download / 'data_mnist', train=True, download=True, transform=mnist_transformer),
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(where_to_download / 'data_mnist', train=False, download=False, transform=mnist_transformer),
        shuffle=True, **kwargs)

    return train_loader, test_loader


def train(model, train_loader, optimizer, epoch, args):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        # loss = distillation(output, target, teacher_output, T=3, alpha=0.5)
        # loss = F.mse_loss(F.softmax(output/3.0), F.softmax(teacher_output/3.0))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '  # len(train_loader)=batch number
                  f'({100. * batch_idx / len(train_loader):.0f}%)\tLoss: {loss.item():.6f}]')


def train_evaluate(model, train_loader, args):
    model.eval()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        train_loss += F.cross_entropy(output, target).item()  # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1]  # select max indices
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.reshape(pred.shape)).cpu().sum()

    print(f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} '
          f'({100. * correct / len(train_loader.dataset):.0f}%)\n')


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max logit
        correct += pred.eq(target.reshape(pred.shape)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
