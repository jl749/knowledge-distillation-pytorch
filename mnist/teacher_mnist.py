from __future__ import print_function
import sys
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path
import threading
import itertools
import time
from utils.base import *
from utils.models import TeacherNet


FILE = Path(__file__).resolve()
BASE_DIR = FILE.parent
done = False


def _animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write(f'\revaluating... ' + c)
        sys.stdout.flush()  # flush buffer
        time.sleep(0.1)
    sys.stdout.write('\rDone!          \n')


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
    global done

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, args)
        done = False; threading.Thread(name='train_evaluate', target=_animate,  daemon=True).start()
        train_evaluate(model, train_loader, args)

        test(model, test_loader, args)
        done = True

    """
    --------------------------------------------------------------------------------------------------------------------
    """
    torch.save(model.state_dict(), 'teacher_MLP.pth.tar')

    print("--- %s seconds ---" % (time.time() - start_time))

    # test_model = Net()
    # test_model.load_state_dict(torch.load('teacher_MLP.pth.tar'))
    #
    # test(test_model, test_loader, args)
    # for data, target in test_loader:
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     teacher_out = test_model(data)
    # print(teacher_out)


if __name__ == '__main__':
    my_args = parse_setting()

    main(my_args)
