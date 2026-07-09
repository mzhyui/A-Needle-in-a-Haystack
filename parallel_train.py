import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
from tqdm import tqdm

import concurrent
from queue import Queue
from threading import Thread
import torch.multiprocessing as mp

from utils.trainUtils import loadModel

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(net, train_loader, criterion, optimizer, epoch, device):
def train(net, train_loader, criterion, optimizer, epoch, device):
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    # print('Epoch %d: %.4f' % (epoch, loss.item()))

@torch.no_grad()
def test(net, train_loader, criterion, optimizer, epoch):
    begin_time = time.time()
    net.eval()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        output = net(data)
        loss = criterion(output, target)
    # print('Epoch %d: %.4f' % (epoch, loss.item()))
    print(time.time() - begin_time)

def train_process(rank, net, train_loader, criterion, optimizer, epoch, device):
    train(net, train_loader, criterion, optimizer, epoch, device=device)
def train_process(rank, net, train_loader, criterion, optimizer, epoch, device):
    train(net, train_loader, criterion, optimizer, epoch, device=device)

def test_process(rank, net, train_loader, criterion, optimizer, epoch):
    test(net, train_loader, criterion, optimizer, epoch)

def trainWorker(task):
    idx, model, dataloader, criterion, optimizer, epoch = task
    model.train()
    loss_list = []
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(loss_list)


def singleP(num_models, device):
    # Define the loss function and optimizer
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()

    # Train the network
    begin = time.time()
    for _ in range(num_models):
        net = LeNet().to(device)
        # net = nn.DataParallel(net)
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        for epoch in range(10):
            train(net, train_loader, criterion, optimizer, epoch, device)

    elapsed = time.time() - begin
    print('Elapsed time: %.2f' % elapsed)

def singleP_reverse(num_models, device):
    # Define the loss function and optimizer
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()

    # Train the network
    begin = time.time()
    for epoch in range(10):
        for _ in range(num_models):
            net = LeNet().to(device)
            # net = nn.DataParallel(net)
            optimizer = optim.SGD(net.parameters(), lr=0.01)
            train(net, train_loader, criterion, optimizer, epoch, device)

    elapsed = time.time() - begin
    print('Elapsed time: %.2f' % elapsed)

def cpuParallel(num_models, threads = 4):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # net_list = [nn.DataParallel(LeNet()) for _ in range(num_models)]
    net_list = [LeNet() for _ in range(num_models)]
    # net_list = [nn.DataParallel(LeNet()) for _ in range(num_models)]
    net_list = [LeNet() for _ in range(num_models)]
    optimizer_list = [optim.SGD(x.parameters(), lr=0.01) for x in net_list]
    train_loader_list = [torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True, num_workers=0) for _ in range(num_models)]
        batch_size=64, shuffle=True, num_workers=0) for _ in range(num_models)]

    # Train the network
    begin = time.time()
    for epoch in range(10):
        # train(net, train_loader, criterion, optimizer, epoch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for i in range(num_models):
                futures.append(executor.submit(train, net_list[i], train_loader_list[i], criterion, optimizer_list[i], epoch, 'cpu'))
                futures.append(executor.submit(train, net_list[i], train_loader_list[i], criterion, optimizer_list[i], epoch, 'cpu'))

            for future in concurrent.futures.as_completed(futures):
                future.result()

    elapsed = time.time() - begin
    print('Elapsed time: %.2f' % elapsed)

def gupParallel(num_models, gpu_list, threads = 4):
def gupParallel(num_models, gpu_list, threads = 4):
    mp.set_start_method('spawn')
    net_list = [nn.DataParallel(LeNet(), gpu_list).cuda() for _ in range(num_models)]
    net_list = [nn.DataParallel(LeNet(), gpu_list).cuda() for _ in range(num_models)]
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_list = [optim.SGD(x.parameters(), lr=0.01) for x in net_list]
    train_loader_list = [torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True, num_workers=threads) for _ in range(num_models)]

    # # Train the network
    begin = time.time()
    pbar = tqdm(range(10))
    for epoch in pbar:
        processes = []
        for i in range(num_models):
            p = mp.Process(target=train_process, args=(i, net_list[i], train_loader_list[i], criterion, optimizer_list[i], epoch, 'cuda'))
            p = mp.Process(target=train_process, args=(i, net_list[i], train_loader_list[i], criterion, optimizer_list[i], epoch, 'cuda'))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    elapsed = time.time() - begin
    print('Elapsed time: %.2f' % elapsed)

def gpuQueueParallel(num_models, threads = 4):
    mp.set_start_method('spawn')
    net_list = [LeNet() for _ in range(num_models)]
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_list = [optim.SGD(x.parameters(), lr=0.01) for x in net_list]
    train_loader_list = [torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True, num_workers=4) for _ in range(num_models)]
    
    process_queue = Queue()
    workers = []
    def worker(process_queue):
        """Worker function to start and manage processes."""
        while True:
            args = process_queue.get()
            if args is None:  # Sentinel value to exit
                break
            i, net, train_loader, criterion, optimizer, epoch = args
            p = mp.Process(target=train_process, args=(i, net, train_loader, criterion, optimizer, epoch))
            p.start()
            p.join()

    begin = time.time()

    for _ in range(threads):
        t = Thread(target=worker, args=(process_queue,))
        t.start()
        workers.append(t)

    # Add tasks to the queue
    for epoch in range(10):
        for i in range(num_models):
            process_queue.put((i, net_list[i], train_loader_list[i], criterion, optimizer_list[i], epoch))

    # Stop all worker threads
    for _ in range(threads):
        process_queue.put(None)
    for t in workers:
        t.join()

    elapsed = time.time() - begin
    print('Elapsed time: %.2f' % elapsed)

def gpuPoolParallel(num_models, threads = 4):
    mp.set_start_method('spawn')
    net_list = [loadModel('resnet20', 'cifar10', 3, 10, pre_trained=False) for _ in range(num_models)]
    optimizer_list = [optim.SGD(x.parameters(), lr=0.01) for x in net_list]
    train_loader_list = [torch.utils.data.DataLoader(
        datasets.CIFAR10('data/cifar10', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=256, shuffle=True, num_workers=0) for _ in range(num_models)]
    
    begin = time.time()
    
    for epoch in range(10):
        with mp.Pool(processes=threads) as pool:
            tasks = [(i, net_list[i], train_loader_list[i], nn.CrossEntropyLoss(), optimizer_list[i], epoch) for i in range(num_models)]
            results = pool.map(trainWorker, tasks)
            for i, loss in enumerate(results):
                print(f'Epoch {epoch} model {i}: {loss}')

    elapsed = time.time() - begin
    print('Elapsed time: %.2f' % elapsed)


def gpuQueueParallelCifar(num_models, threads = 4):

    mp.set_start_method('spawn')
    net_list = [nn.DataParallel(loadModel('resnet20', 'cifar10', 3, 10, pre_trained=False)) for _ in range(num_models)]
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_list = [optim.SGD(x.parameters(), lr=0.01) for x in net_list]
    train_loader_list = [torch.utils.data.DataLoader(
        datasets.CIFAR10('data/cifar10', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=256, shuffle=True, num_workers=4) for _ in range(num_models)]
    
    process_queue = Queue()
    workers = []
    def worker(process_queue):
        """Worker function to start and manage processes."""
        while True:
            args = process_queue.get()
            if args is None:  # Sentinel value to exit
                break
            i, net, train_loader, criterion, optimizer, epoch = args
            p = mp.Process(target=test_process, args=(i, net, train_loader, criterion, optimizer, epoch))
            p.start()
            p.join()

    begin = time.time()

    for _ in range(threads):
        t = Thread(target=worker, args=(process_queue,))
        t.start()
        workers.append(t)

    # Add tasks to the queue
    for epoch in range(1):
        for i in range(num_models):
            process_queue.put((i, net_list[i], train_loader_list[i], criterion, optimizer_list[i], epoch))

    # Stop all worker threads
    for _ in range(threads):
        process_queue.put(None)
    for t in workers:
        t.join()

    elapsed = time.time() - begin
    print('Elapsed time: %.2f' % elapsed)

if __name__ == '__main__':
    # Define the network
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n','--num_models', type=int, default=20, help='number of models')
    argparser.add_argument('-e','--epochs', type=int, default=3, help='number of models')
    argparser.add_argument('-w','--threads', type=int, default=4, help='threads')
    argparser.add_argument('-s', action='store_true', help='single process')
    argparser.add_argument('-sr', action='store_true', help='single process rev')
    argparser.add_argument('-c', action='store_true', help='cpu parallel')
    argparser.add_argument('-p', action='store_true', help='gpu parallel')
    argparser.add_argument('-pp', action='store_true', help='gpu pool parallel') # to be eval
    argparser.add_argument('-q', action='store_true', help='gpu queue parallel')
    argparser.add_argument('--qc', action='store_true', help='gpu queue parallel cifar')
    argparser.add_argument('-d', '--device', type=int, default=-1, help='gpu device id')
    argparser.add_argument('-d', '--device', type=int, default=-1, help='gpu device id')

    args = argparser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >=0 else "cpu")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >=0 else "cpu")

    # Create a network
    if args.s:
        singleP(args.num_models, device)
    gc.collect()
    torch.cuda.empty_cache()
    if args.sr:
        singleP_reverse(args.num_models, device)
    gc.collect()
    torch.cuda.empty_cache()
    if args.c:
        cpuParallel(args.num_models, args.threads)
    if args.c:
        cpuParallel(args.num_models, args.threads)
    gc.collect()
    torch.cuda.empty_cache()
    # TODO 2024-12-12 git.V.9ea50: p function needs to be fixed
    if args.p:
        gupParallel(args.num_models, [0,1], args.threads)
    gc.collect()
    torch.cuda.empty_cache()
    if args.q:
        gpuQueueParallel(args.num_models, args.threads)
    if args.q:
        gpuQueueParallel(args.num_models, args.threads)
    gc.collect()
    torch.cuda.empty_cache()
    if args.pp:
        gpuPoolParallel(args.num_models, args.threads)
    if args.pp:
        gpuPoolParallel(args.num_models, args.threads)
    gc.collect()
    torch.cuda.empty_cache()
    if args.qc:
        gpuQueueParallelCifar(args.num_models, args.threads)
    if args.qc:
        gpuQueueParallelCifar(args.num_models, args.threads)
    gc.collect()
    torch.cuda.empty_cache()