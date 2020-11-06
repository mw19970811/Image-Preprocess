import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import time
import os
import argparse
import torchvision.transforms as transforms
from models.raw_vgg import *

def Args():
    parser = argparse.ArgumentParser(description='image preprocess')
    parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--test-batch-size', type=int, default=20)
    parser.add_argument('--test-num-workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.001, )
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--net-preprocess', type=str, default='bin')
    parser.add_argument('--net-name', type=str, default='vgg16') # raw_vgg16
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    return parser.parse_args()


def adjust_lr(args, optimizer, epoch):
    if epoch in [args.epochs*0.5, args.epochs*0.75, args.epochs*0.85]:
        for p in optimizer.param_groups:
            p['lr'] *= 0.1
            lr = p['lr']
        print('Change lr:'+str(lr))


def train(args, trainloader, model, epoch, optimizer):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    adjust_lr(args, optimizer, epoch)
    for batch_idx, (data, target) in enumerate(trainloader):

        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()

        optimizer.step()
    print('Train Epoch: {}, loss: {:.6f}, acc: {:.4f}%'.format(epoch, loss.item(), 100.*train_acc/len(trainloader.dataset)), end=' | ')


def val(args, testloader, model, epoch):
    model.eval()
    test_loss = 0.
    correct=0.
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(args.device), label.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(output, label, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    test_loss/=len(testloader.dataset)
    correct = int(correct)
    print('Test set:average loss: {:.4f}, accuracy: {:.4f}%'.format(test_loss, 100.*correct/len(testloader.dataset)))
    return correct/len(testloader.dataset)


def main():
    args = Args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    if args.dataset == 'cifar10':
        numclasses=10
        if not os.path.exists('data/CIFAR10'):
            os.mkdir('data/CIFAR10')
        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.Pad(4),
                                                    transforms.RandomCrop(32),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                                                ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.test_num_workers)
    elif args.dataset=='cifar100':
        numclasses=100
        if not os.path.exists('data/CIFAR100'):
            os.mkdir('data/CIFAR100')
        trainset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.Pad(4),
                                                    transforms.RandomCrop(32),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR100(root='./data/CIFAR10', train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = eval(args.net_name)(args.net_preprocess, num_classes=numclasses)
    model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print(str(args))

    best_val_acc=0.
    Save_Path = os.path.join(args.checkpoint,args.net_preprocess+'_'+args.net_name+'_'+str(best_val_acc)+'.pth')
    for i in range(args.epochs):
        train(args, trainloader, model, i+1, optimizer)
        temp_acc = val(args, testloader, model, i+1)
        if temp_acc > best_val_acc:
            if os.path.exists(Save_Path):
                os.remove(Save_Path)
            best_val_acc = temp_acc
            Save_Path = os.path.join(args.checkpoint,args.net_preprocess+'_'+args.net_name+'_'+str(best_val_acc)+'.pth')
            torch.save(model.state_dict(), Save_Path)
        else:
            if os.path.exists(Save_Path):
                model.load_state_dict(torch.load(Save_Path))
    print('Best acc{}'.format(best_val_acc))
    


if __name__ == "__main__":
    main()