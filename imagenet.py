'''Train ImageNet with PyTorch.
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import models as models
from utils import *
import torchvision.datasets as datasets

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r',default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--netName', default='resnet18',choices=model_names, type=str, help='choosing network')
parser.add_argument('--bs', default=1024, type=int, help='batch size')
parser.add_argument('--workers', default=4, type=int, help='workers')
parser.add_argument('--es', default=100, type=int, help='epoch size')
parser.add_argument('--imagenet', default=1000, type=int, help='dataset classes number')
parser.add_argument('--datapath', default='/home/xm0036/Datasets/ImageNet', type=str, help='dataset path')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

# Data loading code
traindir = os.path.join(args.datapath, 'train')
testdir = os.path.join(args.datapath, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_dataset = datasets.ImageFolder(traindir,transform_train)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers)

test_dataset = datasets.ImageFolder(testdir,transform_test)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers)


# Model
print('==> Building model..')
net = models.__dict__[args.netName]()

flops, params = get_model_complexity_info(net, (224, 224),as_strings=True, print_per_layer_stat=False)
print('Flops:  ' + flops)
print('Params: ' + params)


net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_path = './checkpoint/ckpt_'+args.netName+'.t7'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    print("BEST_ACCURACY: "+str(best_acc))
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


# Training
def train(epoch):
    adjust_learning_rate(optimizer, epoch, args.lr)
    print('\nEpoch: %d   Learning rate: %f' % (epoch, optimizer.param_groups[0]['lr']))
    print("\nAllocated GPU memory:", torch.cuda.memory_allocated())
    net.train()
    train_loss = 0
    correct = 0
    correct_1 = 0
    correct_5 = 0
    total = 0



    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted =outputs.topk(5, 1, True, True)
        total += targets.size(0)
        predicted = predicted.t()
        correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
        correct_1_batch = correct[:1].view(-1).float().sum(0, keepdim=True)
        correct_1 += float(correct_1_batch)
        correct_5_batch = correct[:5].view(-1).float().sum(0, keepdim=True)
        correct_5 += float(correct_5_batch)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct_1/total, correct_1, total))

    file_path='records/imagenet_' +args.netName+'_train.txt'
    record_str=str(epoch)+'\t'+"%.3f"%(train_loss/(batch_idx+1))+'\t'+"%.3f"%(100.*correct_1/total)
    +'\t'+"%.3f"%(100.*correct_5/total)+'\n'
    write_record(file_path,record_str)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct_1 = 0
    correct_5 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted =outputs.topk(5, 1, True, True)
            total += targets.size(0)
            predicted = predicted.t()
            correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
            correct_1_batch = correct[:1].view(-1).float().sum(0, keepdim=True)
            correct_1 += float(correct_1_batch)
            correct_5_batch = correct[:5].view(-1).float().sum(0, keepdim=True)
            correct_5 += float(correct_5_batch)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct_1/total, correct_1, total))

    file_path = 'records/imagenet_' +args.netName+ '_test.txt'
    record_str = str(epoch) + '\t' + "%.3f" % (test_loss / (batch_idx + 1)) + '\t' + "%.3f" % (
                100. * correct_1 / total)+'\t'+"%.3f"%(100.*correct_5/total) + '\n'
    write_record(file_path, record_str)

    # Save checkpoint.
    acc = 100.*correct_1/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = './checkpoint/ckpt_' + args.netName + '.t7'
        torch.save(state, save_path)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.es):
    train(epoch)
    test(epoch)


# write statistics to files
statis_path = 'records/STATS_'+args.netName+'.txt'
if not os.path.exists(statis_path):
    # os.makedirs(statis_path)
    os.system(r"touch {}".format(statis_path))
f = open(statis_path, 'w')
statis_str="============\nDivces:"+device+"\n"
statis_str+='\n===========\nargs:\n'
statis_str+=args.__str__()
statis_str+='\n==================\n'
statis_str+="BEST_accuracy: "+str(best_acc)
statis_str+='\n==================\n'
statis_str+="Flops:  "+flops+'\n'
statis_str+="Params:  "+params+'\n'
f.write(statis_str)
f.close()