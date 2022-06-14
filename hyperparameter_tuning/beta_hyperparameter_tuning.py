import sys
import argparse

import torch.cuda
import numpy as np
from common import Logger
from attack import MSAttack
from defense import MSDefense
import common as comm
import matplotlib.pyplot as plt


def test_linf_pgd_attacks():
    print("Testing pgd attack with estimates")
    cuda = True
    # model choose between 'resnet18', 'Net3Conv' and 'Net2Conv'
    model_name = 'net3conv'
    pretrained_path = None
    pretrained_path = '../saved_model/pretrained_net/net3conv_mnist_high_acc.pth'
    # pretrained_path = 'saved_model/pretrained_net/net2conv_mnist.pth'
    # pretrained_path = 'saved_model/pretrained_net/net3conv_f-mnist_low_acc.pth'
    # pretrained_path = 'saved_model/pretrained_net/resnet18_cifar10.pth'

    # Choose dataset 'mnist', 'f-mnist'  not yet addded: 'CIFAR10', 'Imagenet'
    dataset = 'mnist'
    num_class = 10
    test_size = np.arange(0, 500)
    batch_size = 100

    # Learning defence model
    epoch_b = 20

    # learning rate
    lr = 0.0001

    # attack_name: L2/Linf-PGD ,'L2/Linf-MOM' etc.
    attack_name = 'Linf-MOM'

    # pgd attack arguments

    epsilon = 0.3
    eps_iters = 0.025
    nb_iter = 50
    nb_samples = 50

    print("epsilons: {}, eps_iters: {}, nb_iter: {}".format(epsilon, eps_iters, nb_iter))
    # Estimator parameters
    zos = [True, True, True]
    # choose between 'one-point-residual' , 'two-point-forward', 'two-point-backward', 'two-point-central',  Not:'finite-difference' cannot hold memory
    zo_types = ['one-point-residual', 'two-point-forward', 'two-point-central']
    fd_eta = [2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06]

    msd = MSDefense(cuda=cuda, model_name=model_name, dataset=dataset, num_class=num_class, test_size=test_size,
                    batch_size=batch_size, epoch_b=epoch_b, lr=lr)
    msd.load(netv_path=pretrained_path)

    acc_model = comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=True)
    print("Accuracy attacked model: {}".format(acc_model))

    results = []
    for zo, zo_type in zip(zos, zo_types):
        print("Testing: {}".format(zo_type))
        results_type = []
        for fd in fd_eta:
            # print("Epsilon value: {}, eps_iter: {}".format(eps, eps_iter))
            msa = MSAttack(defense_obj=msd, attack_name=attack_name, cuda=cuda, dataset=dataset, num_class=num_class,
                           test_size=test_size,
                           batch_size=batch_size, eps=epsilon, eps_iter=eps_iters, nb_iter=nb_iter, zo=zo,
                           zo_type=zo_type,
                           nb_samples=nb_samples, fd_eta=fd)
            msa.load()
            acc = msa.attack()
            results_type.append(acc)
        results.append(results_type)
        print("Accuracy: {}".format(results_type))
        if zo:
            print("Number of queries used: {}".format(msa.get_number_of_queries()))

    print("Accuracy attacked model: {}".format(acc_model))
    print("Defense model: {}".format(model_name))
    print("Dataset: {}".format(dataset))
    print("Attack_name: {}".format(attack_name))
    print("epsilons: {}".format(epsilon))
    print("eps_iter: {}".format(eps_iters))
    print("nb_iter: {}".format(nb_iter))
    print("nb_samples: {}".format(nb_samples))
    print("fd_eta: {}".format(fd_eta))
    print("All results: ")
    for (t, r) in zip(zo_types, results):
        print("Type: {} , results: {}".format(t, r))
        plt.plot(fd_eta, r, label=t)

    plt.xlabel("epsilons")
    plt.ylabel("Accuracy of attack")
    plt.legend()
    plt.title("Different use of estimates of a {} attack on {} {} model".format(attack_name, dataset, model_name))
    plt.savefig("plots/Hyperparameters/hyper_parameter_tuning_beta_{}_{}_{}.png".format(attack_name, dataset, model_name))
    plt.show()

    return epsilons, results, acc_model


if __name__ == '__main__':
    sys.stdout = Logger('../ms_attack.log', sys.stdout)
    epsilons, results, acc_model = test_linf_pgd_attacks()
