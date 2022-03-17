from dataset import test_set
from predict_function import *

model = torch.load("model_mAcc-0.922_mIoU-0.668.pt")
mob_miou = miou_score(model, test_set)
print('Test Set mIoU', np.mean(mob_miou))
mob_acc = pixel_acc(model, test_set)
print('Test Set acc', np.mean(mob_acc))