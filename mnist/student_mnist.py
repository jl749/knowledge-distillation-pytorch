from __future__ import print_function
import sys
import torch
import torch.optim as optim
from pathlib import Path
import threading
import itertools
import time
from utils.base import *
from utils.models import StudentNet


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
    kwargs = {'num_workers': 1, 'pin_memory': True, 'batch_size': args.batch_size} if args.cuda\
        else {'batch_size': args.batch_size}

    train_loader, test_loader = get_Train_Test_loaders(BASE_DIR, **kwargs)

    """
    init model
    --------------------------------------------------------------------------------------------------------------------
    """
    model = StudentNet()
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
    my_args = parse_setting(FILE.stem)

    main(my_args)

