
import matplotlib.pyplot as plt
import argparse
import logging
from pathlib import Path
import numpy as np
import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from datasets import Voc2012
from networks import Classifier, PAN, ResNet50, Mask_Classifier,ResNet101
from utils import save_model, get_each_cls_iu, PolyLR
from sklearn.metrics import average_precision_score
import ss_transforms as tr

parser = argparse.ArgumentParser(description='PAN')
parser.add_argument('--in_size_h', type=int, default=256,
                    help='input batch size for training (default: 4)')
parser.add_argument('--in_size_w', type=int, default=256,
                    help='input batch size for training (default: 4)')
parser.add_argument('--test_size_h', type=int, default=256,
                    help='input batch size for training (default: 4)')
parser.add_argument('--test_size_w', type=int, default=256,
                    help='input batch size for training (default: 4)')                     
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=120,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=4e-3, help='learning rate (default:4e-3)')
parser.add_argument('--gama', type=float, default=1,
                    help='Cls Loss')
parser.add_argument('--beta', type=float, default=1,
                    help='Semantic Segmentation loss')
parser.add_argument('--ratio', type=float, default=4,
                    help='Semantic Segmentation loss')                   
parser.add_argument('--k1', type=float, default=1,
                    help='Cls Loss')
parser.add_argument('--k2', type=float, default=1,
                    help='Semantic Segmentation loss')
parser.add_argument('--k3', type=float, default=4,
                    help='Semantic Segmentation loss')
                    
parser.add_argument('--bata53', type=float, default=1,
                    help='Semantic Segmentation loss')
parser.add_argument('--bata52', type=float, default=1,
                    help='Semantic Segmentation loss')
parser.add_argument('--bata43', type=float, default=1,
                    help='Semantic Segmentation loss')
parser.add_argument('--bata42', type=float, default=1,
                    help='Semantic Segmentation loss')
parser.add_argument('--bata32', type=float, default=1,
                    help='Semantic Segmentation loss')
                                                             
args = parser.parse_args()
now = datetime.datetime.now()
now_str = 'gau_mul_guid_weight_{}-{}-{}_{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
experiment_name = 'batch_size{:}bata53{:}bata52{:}bata43{:}bata42{:}bata32{:}_{:},{:}_{:},{:}_cfab_ratio_{:}_{:}'.format(args.batch_size, args.bata53,args.bata52,args.bata43,args.bata42,args.bata32, args.in_size_h,args.in_size_w,args.test_size_h,args.test_size_w,args.ratio,now_str)
path_log = Path('./log/' + experiment_name + '.log')

try:
    if path_log.exists():
        raise FileExistsError
except FileExistsError:
    print("Already exist log file: {}".format(path_log))
    raise
else:
    logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                                datefmt='%a, %d %b %Y %H:%M:%S',
                                filename=path_log.__str__(),
                                filemode='w'
                                )
    print('Create log file: {}'.format(path_log))

logging.info('{:}'.format(args))
logging.info("-------------------------------------------------------------------------------")
logging.info("bata53 = {:},bata52 = {:}".format(args.bata53,args.bata52))
logging.info("bata43 = {:},bata42 = {:},bata32 = {:}".format(args.bata43,args.bata42,args.bata32))
logging.info("-------------------------------------------------------------------------------")

train_transforms = transforms.Compose([tr.RandomSized((args.in_size_h, args.in_size_w)),
                                       tr.RandomRotate(15),
                                       tr.RandomHorizontalFlip(),
                                       tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                       tr.ToTensor()
        ])

test_transforms = transforms.Compose([tr.RandomSized((args.test_size_h, args.test_size_w)),
                                      tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                      tr.ToTensor()
])

training_data = Voc2012('/home/gb502/wcl/VOC2012', 'train',transform=train_transforms)
## VOC增强数据集(VOC2012+SBD)
#training_data = Voc2012('/home/gb502/wcl/VOC2012', 'train',transform=train_transforms)
test_data = Voc2012('/home/gb502/wcl/VOC2012', 'val',transform=test_transforms)
training_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

length_training_dataset = len(training_data)
length_test_dataset = len(test_data)

NUM_CLASS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#convnet = ResNet50(pretrained=True)
convnet = ResNet101(pretrained=True)
pan = PAN(convnet.blocks[::-1],args)
mask_classifier = Mask_Classifier(in_features=256, num_class=(NUM_CLASS+1))

convnet.to(device)
pan.to(device)
mask_classifier.to(device)

def train(epoch, optimizer, data_loader):
    convnet.train()
    pan.train()
    pixel_acc = 0
    loss = 0
    
    for batch_idx, (imgs,mask_labels) in enumerate(data_loader):
        imgs, mask_labels = imgs.to(device), mask_labels.to(device)
        fms_blob, z = convnet(imgs)
        out_ss = pan(fms_blob[::-1])
        
        # Semantic Segmentation Loss
        mask_pred = mask_classifier(out_ss)
        #mask_labels = F.interpolate(mask_labels, scale_factor=0.25, mode='nearest')
        pred_b,pred_c,pred_h,pred_w = mask_pred.shape
        mask_labels = F.interpolate(mask_labels,size = (pred_h,pred_w), mode='nearest')
        loss_ss = F.cross_entropy(mask_pred, mask_labels.long().squeeze(1))
        loss = loss + loss_ss

        # Update model
        model_name = [convnet, pan, mask_classifier]
        for m in model_name:
            m.zero_grad()
        #(args.alpha*loss_cls + args.beta*loss_ss).backward()
        loss_ss.backward()
        model_name = ['convnet', 'pan', 'mask_classifier']
        for m in model_name:
            optimizer[m].step()
        # Result
        pixel_acc += mask_pred.max(dim=1)[1].data.cpu().eq(mask_labels.squeeze(1).cpu()).float().mean()

        if (batch_idx+1) % 64 == 0:
            #acc = average_precision_score(np.concatenate(y_true, 0), np.concatenate(y_pred, 0))
            logging.info(
               "Train Epoch:{:}, {:}/{:},   pixel_acc:{:.4f}%, train_loss:{:}".format(
                    epoch, args.batch_size*batch_idx, length_training_dataset, pixel_acc/batch_idx*100,loss/len(data_loader)))
    return loss/len(data_loader)

def test(data_loader):
    global best_acc
    convnet.eval()
    pan.eval()
    all_i_count = []
    all_u_count = []
    y_true = []
    y_pred = []
    pixel_acc = 0
    test_loss = 0
    for batch_idx, (imgs, mask_labels) in enumerate(data_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            mask_labels = mask_labels.to(device)
            fms_blob, z = convnet(imgs)
            out_ss = pan(fms_blob[::-1])
            mask_pred = mask_classifier(out_ss)

        # results
        #mask_labels = F.interpolate(mask_labels, scale_factor=0.25, mode='nearest')
        pred_b,pred_c,pred_h,pred_w = mask_pred.shape
        mask_labels = F.interpolate(mask_labels,size = (pred_h,pred_w), mode='nearest')

        test_loss_ss = F.cross_entropy(mask_pred, mask_labels.long().squeeze(1))
        test_loss = test_loss + test_loss_ss
        #---end ------
        i_count, u_count = get_each_cls_iu(mask_pred.max(1)[1].cpu().data.numpy(), mask_labels.squeeze(1).cpu().data.numpy())

        all_i_count.append(i_count)
        all_u_count.append(u_count)
        pixel_acc += mask_pred.max(dim=1)[1].data.cpu().eq(mask_labels.cpu().squeeze(1).long()).float().mean().item()

    # Result
    each_cls_IOU = (np.array(all_i_count).sum(0) / np.array(all_u_count).sum(0))
    mIOU = each_cls_IOU.mean()
    pixel_acc = pixel_acc / length_test_dataset

    logging.info("Length of test set:{:}  Each_cls_IOU:{:}  test_loss:{:}  mIOU:{:.4f} PA:{:.4f}".format(length_test_dataset, dict(zip(test_data.classes, (100*each_cls_IOU).tolist())), test_loss/len(data_loader),mIOU*100, pixel_acc))

    if mIOU > best_acc:
        logging.info('==>Save model, best mIOU:{:.3f}%'.format(mIOU*100))
        best_acc = mIOU
        state = {'epoch': epoch,
                 'best_acc': best_acc,
                 'convnet': convnet.state_dict(),
                 'pan': pan.state_dict(),             
                 'mask_classifier': mask_classifier.state_dict(),
                 'optimizer': optimizer,
                 }
        save_model(state, directory='./checkpoints', filename=experiment_name+'.pkl')
    return test_loss/len(data_loader),mIOU

model_name = ['convnet', 'pan', 'mask_classifier']
optimizer = {'convnet': optim.SGD(convnet.parameters(), lr=args.lr, weight_decay=1e-4),
             'pan': optim.SGD(pan.parameters(), lr=args.lr, weight_decay=1e-4),
             'mask_classifier': optim.SGD(mask_classifier.parameters(), lr=args.lr, weight_decay=1e-4)}

optimizer_lr_scheduler = {'convnet': PolyLR(optimizer['convnet'], max_iter=args.epochs, power=0.9),
                          'pan': PolyLR(optimizer['pan'], max_iter=args.epochs, power=0.9),
                          'mask_classifier': PolyLR(optimizer['mask_classifier'], max_iter=args.epochs, power=0.9)}

best_acc = 0
x_epoch = []
y_test_loss = []
y_train_loss = []
y_mIoU = []
for epoch in range(args.epochs):
    x_epoch.append(epoch)
    for m in model_name:
        optimizer_lr_scheduler[m].step(epoch)
    logging.info('Epoch:{:}'.format(epoch))
    y_train_loss.append(train(epoch, optimizer, training_loader))
    #if epoch % 1 == 0:
    loss,test_acc = test(test_loader)
    y_test_loss.append(loss)
    y_mIoU.append(test_acc*100)
    '''    
    plt.figure()
    plt.plot(x_epoch,y_train_loss,marker='o',color='r',label = 'train-loss')
    plt.plot(x_epoch,y_test_loss,marker='*',color = 'b',label = 'test-loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.xlabel('loss')
    pigfile = './piture/{}--{}_loss.png'.format(epoch,args.epochs)
    plt.savefig(pigfile)
    plt.close()
    
    plt.figure()
    plt.plot(x_epoch,y_mIoU,marker='o',color='r',label = 'test-mIoU')
    plt.legend()
    plt.xlabel('epoch')
    plt.xlabel('mIou')
    plt.savefig('./piture/mIoU.png')
    plt.close()
    '''
