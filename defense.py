import os.path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T

import saved_model.cifar10_models.resnet
from saved_model.resnet import ResNet18, ResNet152
from saved_model.nets import Net3Conv, Net2Conv, Net4Conv, NetGenMnist, Net6Conv
from saved_model.cifar10_models.googlenet import GoogLeNet


class MSDefense(object):

    def __init__(self, cuda=False, model_name='net3conv', dataset='mnist', num_class=10, test_size=100, batch_size=500,
                 epoch_b=20, lr=0.0001, attack_obj=None):
        super(MSDefense, self).__init__()
        self.cuda = cuda
        self.model_name = model_name
        self.dataset = dataset
        self.num_class = num_class
        self.test_size = test_size
        self.batch_size = batch_size
        self.epoch_b = epoch_b
        self.lr = lr

        self.msa = attack_obj

        self.netV = None
        self.netB_list = []

        self.test_loader = None
        self.train_loader = None
        self.small_test_set = None

        self.map_location = None

    def load(self, pretrained=False, netv_path=None, netb_plist=None):
        """
        Loading nets and datasets
        """

        if self.dataset == 'mnist':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.MNIST(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.MNIST(root='dataset/', train=True, download=True, transform=transform)

            data_list = [i for i in range(6000, 8000)]
            sampler = torch.utils.data.sampler.SubsetRandomSampler(data_list)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, sampler=sampler, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

            self.load_model_mnist(self.model_name)
        elif self.dataset == 'f-mnist':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.FashionMNIST(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.FashionMNIST(root='dataset/', train=True, download=True, transform=transform)

            data_list = [i for i in range(6000, 8000)]
            sampler = torch.utils.data.sampler.SubsetRandomSampler(data_list)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, sampler=sampler, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

            self.load_model_f_mnist(self.model_name)
        elif self.dataset == 'cifar10':
            # mean and std of cifar dataset
            mean = (0.49139968, 0.48215827, 0.44653124)
            std = (0.24703233, 0.24348505, 0.26158768)

            transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

            test_set = torchvision.datasets.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.CIFAR10(root='dataset/', train=True, download=True, transform=transform)

            self.small_test_set = torch.utils.data.DataLoader(torch.utils.data.Subset(test_set, self.test_size),
                                                              batch_size=self.batch_size, shuffle=False,
                                                              num_workers=2, drop_last=True)

            data_list = [i for i in range(6000, 8000)]
            sampler = torch.utils.data.sampler.SubsetRandomSampler(data_list)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, sampler=sampler, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

            self.load_model_cifar10(self.model_name)

        if self.cuda:
            self.netV = self.netV.cuda()
            self.map_location = lambda storage, loc: storage.cuda()
        else:
            self.netV = self.netV.cpu()
            self.map_location = 'cpu'

        if netb_plist is None:
            self.netB_list.append(self.netV)  # for training NetB
        else:
            # for defense evaluation
            self.netB_list = []
            for c in range(len(netb_plist)):
                self.netB_list.append(self.netV)

        self.netV = nn.DataParallel(self.netV)

        # Train part
        if netv_path is not None:
            state_dict = torch.load(netv_path, map_location=self.map_location)
            self.netV.load_state_dict(state_dict)
            self.netV.eval()
        else:
            self.train_netV("saved_model/pretrained_net")
            self.netV.eval()

        for i in range(len(self.netB_list)):
            self.netB_list[i] = nn.DataParallel(self.netB_list[i])

        if netb_plist is not None:
            for i, path in enumerate(netb_plist, 0):
                state_dict = torch.load(path, map_location=self.map_location)
                self.netB_list[i].load_state_dict(state_dict)

        print("Loading \'defense\' is done.")

    def train_netV(self, save_path):
        print("Starting training the victim net V")

        optimizer = torch.optim.Adam(self.netV.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.netV.train()
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epoch_b):
            print("epoch: %d / %d" % (epoch + 1, self.epoch_b))
            for idx, data in enumerate(self.train_loader, 0):
                inputs, labels = data

                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs = inputs.cpu()
                    labels = labels.cpu()

                optimizer.zero_grad()

                outputs = self.netV(inputs)

                loss = criterion(outputs, labels)
                # loss = self.cross_entropy(F.softmax(outputs, dim=1), F.one_hot(labels, 10).float())

                if idx % 100 == 0:
                    print("loss:", loss)

                loss.backward()
                optimizer.step()

        torch.save(self.netV.state_dict(),
                   os.path.join(save_path, "{}_{}.pth".format(self.model_name, self.dataset)))
        print("Finished training of netV")

    def load_model_mnist(self, model_name):
        if model_name == 'resnet18':
            self.netV = ResNet18()
        elif model_name == 'net2conv':
            self.netV = Net2Conv()
        elif model_name == 'net4conv':
            self.netV = Net4Conv()
        elif model_name == 'resnet152':
            self.netV = ResNet152()
        elif model_name == 'net6conv':
            self.netV = Net6Conv()
        elif model_name == 'netgenmnist':
            self.netV = NetGenMnist()
        else:
            self.netV = Net3Conv()

    def load_model_f_mnist(self, model_name):
        if model_name == 'resnet18':
            self.netV = ResNet18()
        elif model_name == 'net2conv':
            self.netV = Net2Conv()
        elif model_name == 'resnet152':
            self.netV = ResNet152()
        elif model_name == 'net4conv':
            self.netV = Net4Conv()
        elif model_name == 'net6conv':
            self.netV = Net6Conv()
        elif model_name == 'netgenmnist':
            self.netV = NetGenMnist()
        else:
            self.netV = Net3Conv()

    def load_model_cifar10(self, model_name):
        if model_name == 'googlenet':
            self.netV = GoogLeNet()
        elif model_name == 'resnet18':
            self.netV = saved_model.cifar10_models.resnet.resnet18(device=self.map_location)
        elif model_name == 'net4conv':
            self.netV = Net4Conv()
        else:
            self.netV = Net3Conv()
