import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import image_folder
import torchvision.transforms as transforms

import os

from torch.autograd import Variable
from vgg_model import vgg16

use_cuda = torch.cuda.is_available()

re_use = 1
best_acc = 0
is_freeze = 1
num_class = 10
batch_size = 16

# Import dataCifar10
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./dataCifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./dataCifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
net = vgg16(pretrained=True)
# Change the fc layers, the model is for cifar10
new_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
    )
net.classifier = new_classifier

if use_cuda:
    net.cuda()
    # This flag allows you to enable the inbuilt cudnn auto-tuner
    # to find the best algorithm to use for your hardware
    cudnn.benchmark = True

# Loss function
cross_entropy = nn.CrossEntropyLoss()

if is_freeze:
    for param in net.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = True

    # This method cannot be used, I don't know why now.
    # for param in net.parameters():
    #     if param not in net.classifier.parameters():
    #         param.requires_grad = False

    optimizer = optim.SGD(net.classifier.parameters(), lr=0.001, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=5e-4)


# Traing
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 10  == 0:
            _, pred = torch.max(outputs, 1)
            correct_num = ((pred == targets).sum()).data[0]

            cur_acc = correct_num / batch_size
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                epoch, (batch_idx+1) * len(inputs), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.data[0], cur_acc))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = cross_entropy(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100. * correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


def save_model():
    pass

for epoch in range(20):
    train(epoch)
    # test(epoch)