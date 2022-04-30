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


def distill_unlabeled(y, teacher_scores, T):
    return F.kl_div(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * T * T


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
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    """
    train
    --------------------------------------------------------------------------------------------------------------------
    """
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, args)
        done = False; threading.Thread(name='train_evaluate', target=_animate, args=(lambda: done,), daemon=True).start()
        train_evaluate(model, train_loader, args)

        test(model, test_loader, args)
        done = True

    for epoch in range(1, args.epochs + 1):
        train([model, teacher_model], train_loader, optimizer, epoch, args, distillation=distill_unlabeled)
        done = False; threading.Thread(name='train_evaluate', target=_animate, args=(lambda: done,), daemon=True).start()

        train_evaluate(model, train_loader, args)
        test(model)

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
