import torch
from pathlib import Path
from utils.base import get_Train_Test_loaders, test
from utils.models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
BASE_DIR = Path(__file__).parent

kwargs = {'num_workers': 1, 'pin_memory': True, 'batch_size': BATCH_SIZE} if torch.cuda.is_available() \
        else {'batch_size': BATCH_SIZE}

_, test_loader = get_Train_Test_loaders(BASE_DIR, **kwargs)


student_model = StudentNet()
student_model.load_state_dict(torch.load('student.pth.tar'))
test(student_model, test_loader)
# for data, target in test_loader:
#     teacher_out = student_model(data)
# print(teacher_out)

teacher_model = TeacherNet()
teacher_model.load_state_dict(torch.load('teacher_MLP.pth.tar'))
test(teacher_model, test_loader)
# for data, target in test_loader:
#     teacher_out = teacher_model(data)
# print(teacher_out)

# TODO: add distilled student
