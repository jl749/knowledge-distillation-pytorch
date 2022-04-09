from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import time
from pathlib import Path


FILE = Path(__file__).resolve()
BASE_DIR = FILE.parent
start_time = time.time()


# Training settings
def parse_setting():
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

    print(FILE.stem, args)

    return args


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)  # flatten (N, C, H, W) --> (N, L)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


def main(args):
    """ main """
    """
    prepare DataLoader
    --------------------------------------------------------------------------------------------------------------------
    """
    mnist_transformer = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # MNIS 1d
                       ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(BASE_DIR/'data_mnist', train=True, download=True, transform=mnist_transformer),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(BASE_DIR/'data_mnist', train=False, download=False, transform=mnist_transformer),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    """
    init model
    --------------------------------------------------------------------------------------------------------------------
    """
    model = Net()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    """
    train
    --------------------------------------------------------------------------------------------------------------------
    """
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, args)
        train_evaluate(model, train_loader, args)
        test(model, test_loader, args)

    """
    --------------------------------------------------------------------------------------------------------------------
    """
    torch.save(model.state_dict(), 'student.pth.tar')

    print("--- %s seconds ---" % (time.time() - start_time))

    # test_model = Net()
    # test_model.load_state_dict(torch.load('student.pth.tar'))
    #
    # test(test_model, test_loader, args)
    # for data, target in test_loader:
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     teacher_out = test_model(data)
    # print(teacher_out)


# criterion = nn.CrossEntropyLoss()
# def kd_cross_entropy(logits, target):
#     loss_data = F.cross_entropy(logits, target)
#     loss_kd = 0.8 * F.cross_entropy(logits/8.0, target/1.01)
#     loss = loss_data + loss_kd
#     return loss

# def distillation(y, labels, teacher_scores, T, alpha):
#     return F.kl_div(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2. * alpha) \
#     + F.cross_entropy(y, labels) * (1. - alpha)

# def distillation(y, labels, teacher_scores, T, alpha):
#     return F.mse_loss(F.softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2. * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


if __name__ == '__main__':
    my_args = parse_setting()

    main(my_args)
