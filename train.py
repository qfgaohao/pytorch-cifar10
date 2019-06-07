import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import logging
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter


import models


parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10 Training Program')
parser.add_argument('-m', '--model', default='resnet18',
                    type=str, help='the model achitecture')
parser.add_argument('--lr', default=0.01, type=float,
                    help='the initial learning rate')
parser.add_argument('--batch-size', '-b', default=32,
                    type=int, help="The training batch size.")
parser.add_argument('--epochs', default=200, type=int,
                    help="The total number of training epochs")

parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')

parser.add_argument('--scheduler', default="cosine", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

parser.add_argument('--milestones', default="80,120,140,180", type=str,
                    help="milestones for MultiStepLR")


args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
writer = SummaryWriter()


def train_epoch(loader, loss, net, optimizer, device, epoch=-1):
    net.train()
    running_loss = 0.0
    num = 0
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num += labels.size(0)

        if (epoch == 0 or epoch == -1) and i == 0:
            writer.add_graph(net, inputs)

    avg_loss = running_loss / num
    logging.info(f'[Epoch {epoch}] loss: {avg_loss:.3f}')
    writer.add_scalar("Loss", avg_loss, global_step=epoch)


def eval(loader, net, device, epoch=-1):
    net.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            grid = torchvision.utils.make_grid(inputs)
            writer.add_image('images', grid, 0)

    correct = sum(class_correct)
    total = sum(class_total)
    print(
        f'Accuracy of the network on the 10000 test images: {correct / total:.3f}')
    writer.add_scalar(f"Accuracy", correct / total, global_step=epoch)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        writer.add_scalar(f"{classes[i]}_accuracy", class_correct[i] / class_total[i],
        global_step=epoch)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)
    net = getattr(models, args.model)(num_classes=num_classes)
    net.to(device)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                              shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    for epoch in range(args.epochs):
        train_epoch(train_loader, criterion, net, optimizer, device, epoch)
        scheduler.step()
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            eval(test_loader, net, device, epoch)
