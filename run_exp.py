import sys
import argparse

import torch.cuda
import numpy as np
from common import Logger
from attack import MSAttack
from defense import MSDefense
import common as comm


def run_attack():
    cuda = True
    # model choose between 'resnet18', 'Net3Conv' and 'Net2Conv'
    model_name = 'net3conv'
    # pretrained_path = None
    pretrained_path = 'saved_model/pretrained_net/net3conv_mnist_high_acc.pth'
    # 'saved_model/pretrained_net/resnet18_cifar10.pth'

    # Choose dataset 'MNIST',  not yet addded: 'CIFAR10', 'Imagenet'
    dataset = 'mnist'
    num_class = 10
    test_size = np.arange(0, 100)
    batch_size = 10

    # Learning defence model
    epoch_b = 20

    # attack_name: L2/Linf-PGD , CW etc.
    attack_name = 'Linf-PGD'

    # pgd attack arguments
    eps = 0.3
    eps_iter = 0.025
    nb_iter = 50


    # Estimator parameters
    zo = False
    # choose between 'one-point-residual' , 'two-point-forward', 'two-point-backward', 'two-point-central', 'finite-difference'
    zo_type = 'white-box'
    nb_samples = 1
    fd_eta = 0.001

    # learning rate
    lr = 0.0001

    msd = MSDefense(cuda=cuda, model_name=model_name, dataset=dataset, num_class=num_class, test_size=test_size,
                    batch_size=batch_size, epoch_b=epoch_b, lr=lr)

    msd.load(netv_path=pretrained_path)

    msa = MSAttack(defense_obj=msd, attack_name=attack_name, cuda=cuda, dataset=dataset, num_class=num_class, test_size=test_size,
                    batch_size=batch_size, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, zo=zo, zo_type=zo_type, nb_samples=nb_samples, fd_eta=fd_eta)

    msa.load()

    print(msa.attack())

    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=True)
    # msa.attack("BIM")
    # msa.attack("CW")
    # msa.attack("PGD")


if __name__ == '__main__':
    sys.stdout = Logger('attack.log', sys.stdout)
    run_attack()


    # args = argparse.ArgumentParser()
    # print(torch.cuda.is_available())
    # # use gpu or cpu
    # args.add_argument('--cuda', default=True, action='store_true', help='using cuda')
    #
    # # model choose between 'resnet18', 'Net3Conv' and 'Net2Conv'
    # args.add_argument('--model_name', default='net3conv', help='choosing model')
    #
    # # Choose dataset 'MNIST',  not yet addded: 'CIFAR10', 'Imagenet'
    # args.add_argument('--dataset', type=str, default='mnist')
    # args.add_argument('--num_class', type=int, default=10)
    # args.add_argument('--test_size', type=int, default=np.arange(0, 100),
    #                   help="How much is being tested for accuracy of attack")
    # args.add_argument('--batch_size', type=int, default=10, help="How much is being tested for accuracy of attack")
    #
    # # learning defence model
    # args.add_argument('--epoch_b', type=int, default=1, help='for training net V')
    # # args.add_argument('--epoch_g', type=int, default=5, help='for training net S')
    #
    # # pgd attack arguments
    # args.add_argument('--eps', type=float, default=0.3, help='maximum distortion in image')
    # args.add_argument('--eps_iter', type=float, default=0.025, help='for attacking net v')
    # args.add_argument('--nb_iter', type=int, default=50, help='number of iterations')
    #
    # # Zo settings
    # args.add_argument('--zo', default=True, help='using zo')
    # # choose between 'one-point-residual' , 'two-point-forward', 'two-point-backward', 'two-point-central', 'finite-difference'
    # args.add_argument('--zo_type', default='two-point-central', help='using zo')
    # args.add_argument('--nb_samples', default=50, help='num of samples')
    # args.add_argument('--fd_eta', type=float, default=0.001, help='num of samples')
    #
    # # learning rate optimiser
    # args.add_argument('--lr', type=float, default=0.0001)
    # args = args.parse_args()
