import sys
import argparse

import torch.cuda
import numpy as np
from common import Logger
from attack import MSAttack
from defense import MSDefense
import common as comm
import matplotlib.pyplot as plt


def hyperparameter_tuning_alpha():
    print("Testing hyper parameter alpha")
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
    attack_name = 'L2-PGD'

    # pgd attack arguments

    epsilon = 3.
    rates = [7., 7.5 ,8.,8.5]
    eps_iters = [r*epsilon/100 for r in rates]
    nb_iter = 100
    nb_samples = 50

    print("epsilons: {}, eps_iters: {}, nb_iter: {}".format(epsilon, eps_iters, nb_iter))
    # Estimator parameters
    zos = [True, True, True]
    # choose between 'one-point-residual' , 'two-point-forward', 'two-point-backward', 'two-point-central',  Not:'finite-difference' cannot hold memory
    zo_types = ['one-point-residual', 'two-point-forward', 'two-point-central']
    fd_eta = 1.5

    msd = MSDefense(cuda=cuda, model_name=model_name, dataset=dataset, num_class=num_class, test_size=test_size,
                    batch_size=batch_size, epoch_b=epoch_b, lr=lr)
    msd.load(netv_path=pretrained_path)

    acc_model = comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=True)
    print("Accuracy attacked model: {}".format(acc_model))

    results = []
    for zo, zo_type in zip(zos, zo_types):
        print("Testing: {}".format(zo_type))
        results_type = []
        for eps_iter in eps_iters:
            # print("Epsilon value: {}, eps_iter: {}".format(eps, eps_iter))
            msa = MSAttack(defense_obj=msd, attack_name=attack_name, cuda=cuda, dataset=dataset, num_class=num_class,
                           test_size=test_size,
                           batch_size=batch_size, eps=epsilon, eps_iter=eps_iter, nb_iter=nb_iter, zo=zo,
                           zo_type=zo_type,
                           nb_samples=nb_samples, fd_eta=fd_eta)
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
        plt.plot(rates, r, label=t)

    plt.xlabel("Alpha")
    plt.ylabel("Accuracy of attack")
    plt.legend()
    plt.title("Parameter tuning  of alpha with {} attack on {} {} model".format(attack_name, dataset, model_name))
    plt.savefig("plots/Hyperparameters/hyper_parameter_tuning_beta_{}_{}_{}.png".format(attack_name, dataset, model_name))
    plt.show()

    return epsilon, results, acc_model


if __name__ == '__main__':
    sys.stdout = Logger('../ms_attack.log', sys.stdout)
    epsilons, results, acc_model = hyperparameter_tuning_alpha()
