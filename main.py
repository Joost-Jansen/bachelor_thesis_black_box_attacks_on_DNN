import sys
import numpy as np
from common import Logger
from attack import MSAttack
from defense import MSDefense
import common as comm
from plots.plotting import double_plot, single_plot, calc_test_statistic, image_plot


def test_attacks():
    print("Testing pgd attack with estimates")
    cuda = True
    # get example image at epsilon. If None: no example images
    example_image_epsilon = None
    example_indexes_mnist = [3, 2, 1, 30, 4, 23, 11, 0, 18, 7]
    example_indexes_f_mnist = [0, 1, 3, 5, 8, 13, 16, 18, 22, 35]
    example_indexes = [3]  # example_indexes_mnist
    # 0 T - shirt / top 1
    # 1 Trouser 16
    # 2 Pullover 5
    # 3 Dress 3
    # 4 Coat 22
    # 5 Sandal 13
    # 6 Shirt 18
    # 7 Sneaker 0
    # 8 Bag 35
    # 9 Ankle boot 8

    # model choose between 'resnet18', 'Net3Conv' and 'Net2Conv'
    model_name = 'net3conv'
    # pretrained_path = None
    # pretrained_path = 'saved_model/pretrained_net/net4conv_f-mnist.pth'
    pretrained_path = 'saved_model/pretrained_net/net3conv_mnist_high_acc.pth'
    # pretrained_path = 'saved_model/pretrained_net/net2conv_mnist.pth'
    # pretrained_path = 'saved_model/pretrained_net/net2conv_f-mnist.pth'
    # pretrained_path = 'saved_model/pretrained_net/net3conv_f-mnist_low_acc.pth'
    # pretrained_path = 'saved_model/pretrained_net/resnet18_cifar10.pth'

    # Choose dataset 'mnist', 'f-mnist'  not yet addded: 'CIFAR10', 'Imagenet'
    dataset = 'mnist'
    num_class = 10
    number_of_tests = 500
    test_size = np.arange(0, number_of_tests)
    batch_size = 250

    # The number of times the test is repeated
    number_repeat_test = 3

    # Learning defence model
    epoch_b = 20

    # learning rate
    lr = 0.0001

    # query until succes
    query_until_succes = True

    # attack_name: L2/Linf-PGD ,'L2/Linf-MOM' CW etc.
    attack_name = 'L2-PGD'

    # pgd attack arguments
    if attack_name == 'L2-PGD' or attack_name == 'L2-MOM':
        epsilons = [0.05, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        # epsilons = [example_image_epsilon]
        eps_iters = [2.5 * eps / 50 for eps in epsilons]
        nb_iter = 100
        # index of test z value
        test_z_eps = 5
        # after this query average is calculated
        avg_query_eps = 2.5
    elif attack_name == 'Linf-PGD' or attack_name == 'Linf-MOM':
        epsilons = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                    1.0]
        # epsilons = [example_image_epsilon]
        eps_iters = [0.025 for eps in epsilons]
        nb_iter = 50
        # index of test z value
        test_z_eps = 9
        # after this query average is calculated
        avg_query_eps = 0.25
    else:
        epsilons = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        eps_iters = [0.025 for eps in epsilons]
        nb_iter = 50
        # index of test z value
        test_z_eps = 10
        # after this query average is calculated
        avg_query_eps = 0.25

    print("epsilons: {}, eps_iters: {}, nb_iter: {}".format(epsilons, eps_iters, nb_iter))
    # Estimator parameters
    zos = [False, True, True, True, True]
    # choose between 'one-point-residual' , 'two-point-forward', 'two-point-backward', 'two-point-central',  Not:'finite-difference' cannot hold memory
    zo_types = ['white-box', 'two-point-forward', 'two-point-backward', 'two-point-central', 'one-point-residual']
    nb_samples = 50
    fd_eta = 1.5

    msd = MSDefense(cuda=cuda, model_name=model_name, dataset=dataset, num_class=num_class, test_size=test_size,
                    batch_size=batch_size, epoch_b=epoch_b, lr=lr)
    msd.load(netv_path=pretrained_path)

    acc_model = comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=True)
    print("Accuracy attacked model: {}".format(acc_model))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    epsilons2 = []
    results = []
    results_var = []
    results_avg = []
    images = []
    for zo, c, zo_type in zip(zos, colors, zo_types):
        print("Testing: {}".format(zo_type))
        results_type = []
        result_avg_type = []
        results_var_type = []
        for eps, eps_iter in zip(epsilons, eps_iters):
            # print("Epsilon value: {}, eps_iter: {}".format(eps, eps_iter))
            acc, var, avg_queries = 0, 0, 0
            # Repeat x times for avg randomness
            for i in range(number_repeat_test):
                msa = MSAttack(defense_obj=msd, attack_name=attack_name, cuda=cuda, dataset=dataset,
                               num_class=num_class,
                               test_size=test_size,
                               batch_size=batch_size, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, zo=zo,
                               zo_type=zo_type,
                               nb_samples=nb_samples, fd_eta=fd_eta, query_until_succes=query_until_succes,
                               example_image_epsilon=example_image_epsilon)
                msa.load()

                t_acc, t_var, t_avg_queries = msa.attack()
                acc += t_acc / number_repeat_test
                var += t_var / number_repeat_test
                avg_queries += t_avg_queries / number_repeat_test

            results_type.append(round(acc, 3))
            results_var_type.append(round(var, 3))
            if eps > avg_query_eps:
                result_avg_type.append(round(avg_queries, 3))
                if zo_type == zo_types[0]:
                    epsilons2.append(eps)

            if example_image_epsilon is eps:
                adv_images = msa.example_image(example_indexes)
                images.append((zo_type, eps, adv_images))

        if query_until_succes:
            results.append((zo_type, c, results_var_type, results_type))
            results_avg.append((zo_type, c, results_var_type, result_avg_type))
            results_var.append(results_var_type)
        else:
            results.append((zo_type, results_var_type, results_type))
            results_avg.append((zo_type, results_var_type, result_avg_type))
            results_var.append(results_var_type)

        print("Accuracy: {}".format(results_type))
        print("Variance: {}".format(results_var_type))
        if query_until_succes:
            print("Average number of queries until succes: {}".format(result_avg_type))
        if zo:
            print()  # print("Number of queries used: {}".format(msa.get_number_of_queries()))

    if example_image_epsilon:
        title = "{}_{}_{}".format(attack_name, dataset, model_name)
        image_plot(title, images)

    print("Accuracy attacked model: {}".format(acc_model))
    print("Defense model: {}".format(model_name))
    print("Dataset: {}".format(dataset))
    print("Attack_name: {}".format(attack_name))
    print("epsilons: {}".format(epsilons))
    print("eps_iter: {}".format(eps_iters))
    print("nb_iter: {}".format(nb_iter))
    print("nb_samples: {}".format(nb_samples))
    print("fd_eta: {}".format(fd_eta))
    print("All results: ")

    if query_until_succes:
        for (n, c, v, r) in results:
            print("Type {} accuracy : {}".format(n, r))
            print("Type {} variance : {}".format(n, v))

        for (n, c, v, a) in results_avg:
            print("Type {} avg_queries_until_success : {}".format(n, a))

        title = "{} attack on {} {} model".format(attack_name, dataset, model_name)
        x_axis = 'epsilons'
        y_axis1 = 'accuracy of attack'
        y_axis2 = 'average queries until success'
        double_plot("plots", title, x_axis, y_axis1, y_axis2, epsilons, epsilons2, results, results_avg)
    else:
        for (n, v, r) in results:
            print("Type {} accuracy : {}".format(n, r))
            print("Type {} variance : {}".format(n, v))
        title = "{} attack on {} {} model".format(attack_name, dataset, model_name)
        x_axis = 'epsilons'
        y_axis1 = 'accuracy of attack'
        single_plot("plots", title, x_axis, y_axis1, epsilons, results)

    # Calculates Z for two side 95% significance for maintaining accuracy
    #  H0= p1 /= p2 vs. H_1 = p1==p2
    # returns z and bool. bool is true when |z| < z_a/2. Where a is the 95% confidence level
    if query_until_succes:
        (n1, c1, v1, r1) = results[len(results) - 1]
        print("test z at epsilon: {}".format(epsilons[test_z_eps]))
        for (n2, c2, v2, r2) in results:
            z, z_bool = calc_test_statistic(r1[test_z_eps], r2[test_z_eps], number_of_tests)
            print("{}, {} z_test: {}, H_0 rejected: {}".format(n1, n2, z, z_bool))

    else:
        (n1, v1, r1) = results[len(results) - 1]
        print("test z at epsilon: {}".format(epsilons[test_z_eps]))
        for (n2, v2, r2) in results:
            z, z_bool = calc_test_statistic(r1[test_z_eps], r2[test_z_eps], number_of_tests)
            print("{}, {} z_test: {}, H_0 rejected: {}".format(n1, n2, z, z_bool))

    return epsilons, results, acc_model


if __name__ == '__main__':
    sys.stdout = Logger('attack.log', sys.stdout)
    epsilons, results, acc_model = test_attacks()
