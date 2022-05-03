from __future__ import print_function
from pathlib import Path
import threading
import time

import torch
import torch.optim as optim
from utils.base import *
from utils.models import *


FILE = Path(__file__).resolve()
BASE_DIR = FILE.parent


def distillation(y, labels, teacher_scores, T=20.0, alpha=0.7):
    return F.kl_div(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (T*T * 2.0 * alpha)\
           + F.cross_entropy(y, labels) * (1. - alpha)


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
    teacher_model = TeacherNet()
    teacher_model.load_state_dict(torch.load('teacher_MLP.pth.tar'))

    model = StudentNet()
    if args.cuda:
        teacher_model.cuda()
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    """
    train
    --------------------------------------------------------------------------------------------------------------------
    """
    for epoch in range(1, args.epochs + 1):
        train([model, teacher_model], train_loader, optimizer, epoch, args, distillation=distillation)
        done = False; threading.Thread(name='train_evaluate', target=_animate, args=(lambda: done,), daemon=True).start()

        train_evaluate(model, train_loader, args)
        test(model, test_loader, cuda=args.cuda)
        done = True

    """
    --------------------------------------------------------------------------------------------------------------------
    """
    torch.save(model.state_dict(), 'distill.pth.tar')
    # the_model = Net()
    # the_model.load_state_dict(torch.load('student.pth.tar'))

    # test(the_model)
    # for data, target in test_loader:
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     teacher_out = the_model(data)
    # print(teacher_out)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    my_args = parse_setting()

    main(my_args)
