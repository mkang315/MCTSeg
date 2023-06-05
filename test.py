import argparse
import os

import torch
import setproctitle

from model.net import Model
from predict import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--user', default='lizy', type=str)
parser.add_argument('--gpu', default='1', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
             [True, False, False, False],
             [False, True, False, True], [False, True, True, False], [True, False, True, False],
             [False, False, True, True], [True, False, False, True], [True, True, False, False],
             [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
             [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair',
                 't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                 'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                 'flairt1cet1t2']

    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = './datasets/BraTS20/Test_npy'
    test_file = 'test3.txt'
    resume = './runs/BraTS20/output2/model_last.pth'
    num_cls = 4
    dataname = 'BraTS20'

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    # for name, param in model.named_parameters():
    #     print(name, param.device)
    # for name, buffer in model.named_buffers():
    #     print(name, buffer.device)

    # model = model.cuda()
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                test_loader,
                model,
                dataname=dataname,
                feature_mask=mask)
            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))