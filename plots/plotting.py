import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def image_plot(title, images):
    org_added = False
    lists = []
    for i in range(len(images[0][2])):
        lists.append([])

    for (zo_type, eps, adv_images) in images:
        i = 0
        for (org, adv, label, pred) in adv_images:
            if not org_added:
                lists[i].append(org)

            lists[i].append(adv)
            i += 1

        org_added = True

    fig, axs = plt.subplots(len(adv_images), len(images) + 1, figsize=(15, 15))
    axs = axs.flatten()
    axs[0].set_title('original\nimage', size=25)
    axs[1].set_title('white-box\n attack',  size=25)
    axs[3].set_title('two-point\n forward',  size=25)
    axs[4].set_title('two-point\n backward',  size=25)
    axs[2].set_title('two-point\n central',  size=25)
    axs[5].set_title('one-point\n residual', size=25)

    flatten_list = np.concatenate(lists)
    for img, ax in zip(flatten_list, axs):
        ax.axis('off')
        ax.imshow(img, cmap='gray')

    fig.tight_layout()
    plt.savefig("plots/Example_images/7_estimates_{}".format(title))
    plt.show()

def double_plot(path, t, x_axis, y_axis1, y_axis2, eps1, eps2, list1, list2):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel(x_axis)
    ax1.set_ylabel(y_axis1)
    for (l, c, v, r) in list1:
        ax1.plot(eps1, r, label=l, color=c)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel(y_axis2)  # we already handled the x-label with ax1
    for (l, c, v, r) in list2:
        ax2.plot(eps2, r, '--', label=l, color=c)

    ax2.tick_params(axis='y')

    ax2.spines['right'].set_linestyle((0, (8, 5)))
    ax1.spines['right'].set_linestyle((0, (8, 5)))

    plt.title(t)
    # ax2.legend(bbox_to_anchor=(1, 0.8), loc='center right', framealpha=1)
    ax1.legend(bbox_to_anchor=(1, 0.75), loc='center right', framealpha=1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("{}/{}_double_plot".format(path, t))
    plt.show()


def single_plot(path, t, x_axis, y_axis, eps, results):
    for (l, v, r) in results:
        plt.plot(eps, r, label=l)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(t)
    plt.legend()
    plt.savefig("{}/{}_single_plot".format(path, t))
    plt.show()


def parameter_tuning_stats():
    # Linf PGD steps scaling
    # nb_iter = [10,20,30,40,50,75,100]
    # opr_iter = [0.026, 0.016, 0.024, 0.02, 0.012, 0.018, 0.02]
    # tpf_iter =[0.278, 0.782, 0.932, 0.966, 0.97, 0.986, 0.99]
    # nblist = [("two-point-forward", tpf_iter), ("one-point-residual", opr_iter)]
    # title = "Hyperparameter Steps scaling of PGD for different estimates"
    # x_axis = 'Number of steps'
    # y_axis = 'accuracy of attack'
    # single_plot(title, x_axis,y_axis, nb_iter, nblist)

    # Beta hyper parameter tuning
    fd_eta = [5.0, 4.0, 3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06]
    opr_fd = [0.86, 0.86, 0.874, 0.874, 0.866, 0.88, 0.864, 0.886, 0.878, 0.874, 0.878, 0.864, 0.826, 0.77, 0.666, 0.498, 0.074, 0.016, 0.008, 0.008, 0.008]
    tpf_fd = [0.904, 0.918, 0.93, 0.934, 0.938, 0.932, 0.934, 0.934, 0.938, 0.938, 0.934, 0.94, 0.944, 0.938, 0.948, 0.944, 0.938, 0.946, 0.944, 0.754, 0.042]
    tpc_fd = [0.938, 0.942, 0.946, 0.944, 0.946, 0.942, 0.95, 0.946, 0.944, 0.946, 0.944, 0.942, 0.946, 0.952, 0.946, 0.942, 0.94, 0.942, 0.938, 0.862, 0.118]
    fdlist = [('two-point-central', [], tpc_fd), ('two-point-forward', [], tpf_fd), ('one-point-residual', [], opr_fd)]
    title = "Hyperparameter Beta scaling for different estimates"
    x_axis = 'Beta'
    y_axis = 'accuracy of attack'
    single_plot("Hyperparameters", title, x_axis, y_axis, fd_eta, fdlist)
    plt.legend(prop={'size': 15})

    # Alpha hyper parameter tuning
    alpha = [5.0, 4.0, 3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001,
             1e-05, 1e-06]
    opr_alpha = [0.86, 0.86, 0.874, 0.874, 0.866, 0.88, 0.864, 0.886, 0.878, 0.874, 0.878, 0.864, 0.826, 0.77, 0.666,
                 0.498, 0.074, 0.016, 0.008, 0.008, 0.008]
    tpf_alpha = [0.904, 0.918, 0.93, 0.934, 0.938, 0.932, 0.934, 0.934, 0.938, 0.938, 0.934, 0.94, 0.944, 0.938, 0.948,
                 0.944, 0.938, 0.946, 0.944, 0.754, 0.042]
    tpc_alpha = [0.938, 0.942, 0.946, 0.944, 0.946, 0.942, 0.95, 0.946, 0.944, 0.946, 0.944, 0.942, 0.946, 0.952, 0.946,
                 0.942, 0.94, 0.942, 0.938, 0.862, 0.118]
    alphalist = [('two-point-central', [], tpc_alpha), ('two-point-forward', [], tpf_alpha),
                 ('one-point-residual', [], opr_alpha)]
    title = "Hyperparameter alpha scaling for different estimates"
    x_axis = 'alpha'
    y_axis = 'accuracy of attack'
    single_plot("Hyperparameters", title, x_axis, y_axis, alpha, alphalist)
    plt.legend(prop={'size': 15})


def single_plot_stats():
    # L2 PGD
    # eps = [0.05, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    # eps_index = 5

    # results l2 pgd MNIST
    # wb = [0.01, 0.054, 0.2, 0.568, 0.862, 0.968, 0.994, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # opr = [0.01, 0.03, 0.08, 0.2, 0.438, 0.696, 0.878, 0.95, 0.988, 1.0, 1.0, 1.0, 1.0]
    # tpf = [0.01, 0.038, 0.098, 0.306, 0.612, 0.844, 0.938, 0.99, 0.996, 1.0, 1.0, 1.0, 1.0]
    # tpb = [0.01, 0.038, 0.092, 0.296, 0.626, 0.848, 0.936, 0.982, 0.998, 1.0, 1.0, 1.0, 1.0]
    # tpc = [0.01, 0.04, 0.098, 0.316, 0.63, 0.864, 0.944, 0.992, 0.998, 1.0, 1.0, 1.0, 1.0]
    # title = 'Different use of estimates of a L2-PGD attack on MNIST net3conv model'

    # results l2 pgd F-MNIST
    # wb = [0.092, 0.43, 0.788, 0.908, 0.974, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # opr = [0.086, 0.232, 0.566, 0.776, 0.86, 0.924, 0.972, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0]
    # tpf = [0.086, 0.23, 0.544, 0.772, 0.868, 0.94, 0.982, 0.994, 1.0, 1.0, 1.0, 1.0, 1.0]
    # tpb = [0.088, 0.23, 0.556, 0.77, 0.868, 0.934, 0.978, 0.996, 0.998, 1.0, 1.0, 1.0, 1.0]
    # tpc = [0.09, 0.314, 0.666, 0.84, 0.914, 0.974, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # title = 'Different use of estimates of a L2-PGD attack on F-MNIST net3conv model'

    # Linf PGD
    # eps = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # eps_index = 9

    # results linf pgd MNIST
    # wb = [0.01, 0.01, 0.012, 0.018, 0.034, 0.056, 0.258, 0.688, 0.948, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #       1.0]
    # opr = [0.01, 0.01, 0.01, 0.01, 0.012, 0.016, 0.06, 0.17, 0.428, 0.708, 0.906, 0.964, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0,
    #        1.0]
    # tpf = [0.01, 0.01, 0.01, 0.01, 0.014, 0.024, 0.072, 0.252, 0.578, 0.882, 0.97, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #        1.0]
    # tpb = [0.01, 0.01, 0.01, 0.01, 0.014, 0.026, 0.078, 0.254, 0.584, 0.886, 0.972, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #        1.0]
    # tpc = [0.01, 0.01, 0.01, 0.01, 0.014, 0.022, 0.08, 0.26, 0.6, 0.9, 0.978, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # title = 'Different use of estimates of a Linf-PGD attack on MNIST net3conv model'

    # # results linf pgd F-MNIST
    wb = [0.092, 0.096, 0.134, 0.24, 0.416, 0.68, 0.916, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    opr = [0.076, 0.084, 0.088, 0.094, 0.116, 0.19, 0.65, 0.912, 0.984, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    tpf = [0.076, 0.078, 0.086, 0.098, 0.122, 0.218, 0.664, 0.93, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    tpb = [0.076, 0.078, 0.086, 0.094, 0.124, 0.224, 0.652, 0.924, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    tpc = [0.08, 0.084, 0.088, 0.11, 0.138, 0.264, 0.766, 0.958, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    title = 'Different use of estimates of a Linf-PGD attack on F-MNIST net3conv model'

    avg_gap_wb = 100 * abs(sum(wb) - sum(wb)) / len(eps)
    avg_gap_opr = 100 * abs(sum(opr) - sum(wb)) / len(eps)
    avg_gap_tpf = 100 * abs(sum(tpf) - sum(wb)) / len(eps)
    avg_gap_tpb = 100 * abs(sum(tpb) - sum(wb)) / len(eps)
    avg_gap_tpc = 100 * abs(sum(tpc) - sum(wb)) / len(eps)

    print("wb gap: {}".format(avg_gap_wb))
    print("opr gap: {}".format(avg_gap_opr))
    print("tpf gap: {}".format(avg_gap_tpf))
    print("tpb gap: {}".format(avg_gap_tpb))
    print("tpc gap: {}".format(avg_gap_tpc))
    #
    list = [('white-box', [], wb), ('two-point-central', [], tpc), ('two-point-forward', [], tpf),
            ('two-point-forward', [], tpb),
            ('one-point-residual', [], opr)]
    x_axis = 'epsilons (maximum amount of distortion per image)'
    y_axis = 'accuracy of attack'

    for (n2, v2, r2) in list:
        n1, c1, v1, r1 = list[len(list) - 1]
        z, z_bool = calc_test_statistic(r1[eps_index], r2[eps_index])
        print("{}, {} z_test: {}, H_0 rejected: {}".format(n1, n2, z, z_bool))

    single_plot("", title, x_axis, y_axis, list, 500)


def double_plot_stats(linf, mnist):
    if linf:
        eps1 = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                1.0]
        eps2 = [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        eps_index = 9

        if mnist:
            # Linf pgd mnist double plot
            # mnist
            title = 'Linf-PGD attack on MNIST net3conv model'
            wb1 = [0.01, 0.01, 0.012, 0.018, 0.034, 0.056, 0.258, 0.688, 0.948, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0,
                   1.0]
            opr1 = [0.01, 0.01, 0.01, 0.01, 0.012, 0.016, 0.06, 0.17, 0.428, 0.708, 0.906, 0.964, 0.996, 1.0, 1.0, 1.0,
                    1.0,
                    1.0, 1.0]
            tpf1 = [0.01, 0.01, 0.01, 0.01, 0.014, 0.024, 0.072, 0.252, 0.578, 0.882, 0.97, 0.998, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0,
                    1.0]
            tpb1 = [0.01, 0.01, 0.01, 0.01, 0.014, 0.026, 0.078, 0.254, 0.584, 0.886, 0.972, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0,
                    1.0]
            tpc1 = [0.01, 0.01, 0.01, 0.01, 0.014, 0.022, 0.08, 0.26, 0.6, 0.9, 0.978, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0]

            opr2 = [1495.88, 1460.59, 1284.24, 1152.1, 951.7, 754.9, 595.8, 449.6, 216.4]
            tpf2 = [1291.79, 1149.68, 1017.08, 890.66, 722.36, 598.03, 462.06, 343.54, 191.76]
            tpb2 = [1293.32, 1124.31, 1036.83, 904.84, 733.89, 592.42, 458.69, 343.84, 189.62]
            tpc2 = [2544.11, 2172.22, 1919.3, 1703.8, 1383.8, 1153.4, 860.6, 670.2, 375.0]

        else:
            # f-mnist
            title = 'Linf-PGD attack on F-MNIST net3conv model'
            wb1 = [0.092, 0.096, 0.134, 0.24, 0.416, 0.68, 0.916, 0.988, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0]
            opr1 = [0.076, 0.084, 0.088, 0.094, 0.116, 0.19, 0.65, 0.85, 0.906, 0.958, 0.984, 0.996, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0]
            tpf1 = [0.076, 0.078, 0.086, 0.098, 0.122, 0.218, 0.664, 0.852, 0.93, 0.984, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0]
            tpb1 = [0.076, 0.078, 0.086, 0.094, 0.124, 0.224, 0.652, 0.86, 0.932, 0.978, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0]
            tpc1 = [0.08, 0.084, 0.088, 0.11, 0.138, 0.264, 0.766, 0.896, 0.958, 0.992, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0]

            opr2 = [603.72, 559.37, 500.4, 437.3, 350.2, 303.5, 230.6, 207.6, 156.9]
            tpf2 = [548.34, 468.24, 422.89, 380.77, 302.53, 254.29, 202.06, 177.17, 143.82]
            tpb2 = [567.6, 465.63, 421.77, 374.85, 301.92, 251.43, 209.41, 170.95, 130.36]
            tpc2 = [900.18, 806.0, 710.2, 631.4, 542.4, 446.8, 403.0, 331.6, 257.6]

    else:
        # L2 pgd  double plot
        eps1 = [0.05, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        eps2 = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        eps_index = 5

        if mnist:
            title = 'L2-PGD attack on MNIST net3conv model'
            # minst
            wb1 = [0.01, 0.054, 0.2, 0.568, 0.862, 0.968, 0.994, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            opr1 = [0.01, 0.03, 0.08, 0.2, 0.438, 0.696, 0.878, 0.95, 0.988, 1.0, 1.0, 1.0, 1.0]
            tpf1 = [0.01, 0.038, 0.098, 0.306, 0.612, 0.844, 0.938, 0.99, 0.996, 1.0, 1.0, 1.0, 1.0]
            tpb1 = [0.01, 0.038, 0.092, 0.296, 0.626, 0.848, 0.936, 0.982, 0.998, 1.0, 1.0, 1.0, 1.0]
            tpc1 = [0.01, 0.04, 0.098, 0.316, 0.63, 0.864, 0.944, 0.992, 0.998, 1.0, 1.0, 1.0, 1.0]

            opr2 = [4099.16, 3678.62, 3142.79, 2586.7, 2145.7, 1781.1, 1471.8, 1241.0]
            tpf2 = [3559.36, 3021.5, 2495.11, 2010.78, 1668.31, 1354.66, 1122.0, 947.68]
            tpb2 = [3599.2, 3025.6, 2499.53, 2043.98, 1643.22, 1358.54, 1129.75, 943.91]
            tpc2 = [6809.48, 5658.25, 4680.59, 3747.25, 3034.0, 2535.4, 2105.8, 1784.8]

        else:
            title = 'L2-PGD attack on F-MNIST net3conv model'
            # f-mnist
            wb1 = [0.092, 0.43, 0.788, 0.908, 0.974, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            opr1 = [0.086, 0.232, 0.566, 0.776, 0.86, 0.924, 0.972, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0]
            tpf1 = [0.086, 0.23, 0.544, 0.772, 0.868, 0.94, 0.982, 0.994, 1.0, 1.0, 1.0, 1.0, 1.0]
            tpb1 = [0.088, 0.23, 0.556, 0.77, 0.868, 0.934, 0.978, 0.996, 0.998, 1.0, 1.0, 1.0, 1.0]
            tpc1 = [0.09, 0.314, 0.666, 0.84, 0.914, 0.974, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

            opr2 = [1516.77, 1351.33, 1155.8, 894.9, 704.9, 569.0, 467.4, 387.3]
            tpf2 = [1529.59, 1299.58, 989.24, 759.29, 594.56, 474.81, 396.98, 327.93]
            tpb2 = [1580.74, 1260.73, 999.19, 767.04, 600.07, 485.21, 388.31, 321.71]
            tpc2 = [2629.17, 2105.22, 1587.2, 1248.8, 998.0, 825.6, 674.2, 556.2]

    x_axis = 'epsilons'
    y_axis1 = 'accuracy of attack'
    y_axis2 = 'average queries until success'

    list1 = [('white-box', '#1f77b4', [], wb1), ('two-point-central', '#ff7f0e', [], tpc1),
             ('two-point-forward', '#2ca02c', [], tpf1),
             ('two-point-backward', '#d62728', [], tpb1), ('one-point-residual', '#9467bd', [], opr1)]
    list2 = [('two-point-central', '#ff7f0e', [], tpc2), ('two-point-forward', '#2ca02c', [], tpf2),
             ('two-point-backward', '#d62728', [], tpb2), ('one-point-residual', '#9467bd', [], opr2)]

    for (n2, c2, v2, r2) in list1:
        n1, c1, v1, r1 = list1[len(list1) - 1]
        z, z_bool = calc_test_statistic(r1[eps_index], r2[eps_index], 500)
        print("{}, {} z_test: {}, H_0 rejected: {}".format(n1, n2, z, z_bool))

    double_plot("Old", title, x_axis, y_axis1, y_axis2, eps1, eps2, list1, list2)


#  H0= p1 == p2 vs. H_1 = p1<p2
# a= 0.05
# returns z and bool. bool is true when |z| > z_a and the Null hypothesis is rejected. Where a is the 95% confidence level
def calc_test_statistic(p1, p2, number_of_tests):
    p_hat = (p1 + p2)
    if p_hat == 0.:
        return 0, True

    z = round(abs((p1 - p2) / math.sqrt(abs(p_hat * (1 - p_hat) * (1 / number_of_tests)))), 3)
    z_bool = False
    if z > 1.95996:
        z_bool = True
    return z, z_bool


if __name__ == "__main__":
    # single_plot_stats()
    # parameter_tuning_stats()
    linf = False
    mnist = True
    # double_plot_stats(linf, mnist)
    parameter_tuning_stats()