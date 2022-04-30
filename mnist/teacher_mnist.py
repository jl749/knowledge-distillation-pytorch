from __future__ import print_function
from pathlib import Path
import threading
import time

import torch
import torch.optim as optim
from utils.base import *
from utils.models import TeacherNet


FILE = Path(__file__).resolve()
BASE_DIR = FILE.parent


def main(args):
    """ main """
    start_time = time.time()
    """
    prepare DataLoader
    --------------------------------------------------------------------------------------------------------------------
    """
    kwargs = {'num_workers': 1, 'pin_memory': True, 'batch_size': args.batch_size} if args.cuda \
        else {'batch_size': args.batch_size}

    train_loader, test_loader = get_Train_Test_loaders(BASE_DIR, **kwargs)

    """
    init model
    --------------------------------------------------------------------------------------------------------------------
    """
    model = TeacherNet()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    """
    train
    --------------------------------------------------------------------------------------------------------------------
    """
    for epoch in range(1, args.epochs + 1):
        train([model], train_loader, optimizer, epoch, args)
        done = False; threading.Thread(name='train_evaluate', target=_animate, args=(lambda: done,), daemon=True).start()
        train_evaluate(model, train_loader, args)

        test(model, test_loader, cuda=args.cuda)
        done = True

    """
    --------------------------------------------------------------------------------------------------------------------
    """
    torch.save(model.state_dict(), 'teacher_MLP.pth.tar')

    print("--- %s seconds ---" % (time.time() - start_time))

    # test_model = Net()
    #
    #
    # test(test_model, test_loader, args)
    # for data, target in test_loader:
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     teacher_out = test_model(data)
    # print(teacher_out)


if __name__ == '__main__':
    my_args = parse_setting()

    main(my_args)
