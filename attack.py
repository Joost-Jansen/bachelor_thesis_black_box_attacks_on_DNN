import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
import math
from advertorch.attacks import LinfBasicIterativeAttack, GradientSignAttack, LinfPGDAttack, L2PGDAttack, L1PGDAttack, \
    L2MomentumIterativeAttack, LinfMomentumIterativeAttack
from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack
from advertorch.attacks.blackbox.iterative_gradient_approximation import NESAttack
from estimators.gradient_estimate_pgd import gradient_estimator_wrapper
from estimators.gradient_estimate_MIA import gradient_estimator_wrapper_mia


class MSAttack(object):

    def __init__(self, defense_obj=None, attack_name='Linf-PGD', cuda=False, dataset='mnist', num_class=10,
                 test_size=np.arange(0, 500), batch_size=500, eps=0.3, eps_iter=0.025, nb_iter=50, zo=False,
                 zo_type='two-point-residual', nb_samples=50,
                 fd_eta=0.001, query_until_succes=False, example_image_epsilon=None):

        self.msd = defense_obj
        self.attack_name = attack_name
        self.adversary = None

        self.test_set = None
        self.train_loader = None
        self.test_loader = None
        self.cuda = cuda
        self.dataset = dataset
        self.num_class = num_class
        self.test_size = test_size
        self.batch_size = batch_size
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.zo = zo
        self.zo_type = zo_type
        self.nb_samples = nb_samples
        self.fd_eta = fd_eta
        self.query_until_succes = query_until_succes
        self.number_of_tests = self.test_size[len(test_size)-1]

        if zo_type == 'one-point-residual':
            self.number_of_queries = nb_iter * nb_samples
        elif zo_type == 'two-point-forward':
            self.number_of_queries = nb_iter * nb_samples + nb_samples
        elif zo_type == 'two-point-backward':
            self.number_of_queries = nb_iter * nb_samples + nb_samples
        elif zo_type == 'two-point-central':
            self.number_of_queries = nb_iter * nb_samples * 2
        elif zo_type == 'finite-difference':
            self.number_of_queries = nb_iter * 752
        else:
            self.number_of_queries = 0

    def load(self):
        """
        Loading nets and datasets
        """
        if self.dataset == 'mnist':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            self.test_set = torchvision.datasets.MNIST(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.MNIST(root='dataset/', train=True, download=True, transform=transform)

            self.small_test_set = torch.utils.data.DataLoader(torch.utils.data.Subset(self.test_set, self.test_size),
                                                              batch_size=self.batch_size, shuffle=False,
                                                              num_workers=2, drop_last=True)

            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=500, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

            # print('Loading \'attack\' is done.')
        elif self.dataset == "cifar10":
            # mean and std of cifar dataset
            mean = (0.49139968, 0.48215827, 0.44653124)
            std = (0.24703233, 0.24348505, 0.26158768)

            transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

            self.test_set = torchvision.datasets.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.CIFAR10(root='dataset/', train=True, download=True, transform=transform)

            self.small_test_set = torch.utils.data.DataLoader(torch.utils.data.Subset(self.test_set, self.test_size),
                                                              batch_size=self.batch_size, shuffle=False,
                                                              num_workers=2, drop_last=True)

            data_list = [i for i in range(6000, 8000)]
            sampler = torch.utils.data.sampler.SubsetRandomSampler(data_list)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=500, sampler=sampler, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

        elif self.dataset == "f-mnist":
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            train_set = torchvision.datasets.FashionMNIST(root='dataset/', train=False, download=True,
                                                          transform=transform)
            self.test_set = torchvision.datasets.FashionMNIST(root='dataset/', train=True, download=True,
                                                         transform=transform)

            self.small_test_set = torch.utils.data.DataLoader(torch.utils.data.Subset(self.test_set, self.test_size),
                                                              batch_size=self.batch_size, shuffle=False,
                                                              num_workers=2, drop_last=True)

            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=500, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

    def get_adversary(self, method):
        # Estimate type zo
        if self.zo and self.zo_type is not None:
            # print("ZO method")
            if self.zo == 'L2-MOM' or self.zo == 'Linf-MOM':
                adversary = gradient_estimator_wrapper_mia(attack_type=method, predicts=self.msd.netV,
                                                           loss_fns=nn.CrossEntropyLoss(reduction="sum"),
                                                           epss=self.eps, nb_iters=self.nb_iter,
                                                           eps_iters=self.eps_iter,
                                                           clip_mins=0.0, clip_maxs=1.0, targeteds=False,
                                                           nb_sampless=self.nb_samples, fd_etas=self.fd_eta,
                                                           estimator_type=self.zo_type,
                                                           query_until_success=self.query_until_succes)
            else:
                adversary = gradient_estimator_wrapper(attack_type=method, predicts=self.msd.netV,
                                                       loss_fns=nn.CrossEntropyLoss(reduction="sum"),
                                                       epss=self.eps, nb_iters=self.nb_iter,
                                                       eps_iters=self.eps_iter,
                                                       rand_inits=True, clip_mins=0.0, clip_maxs=1.0, targeteds=False,
                                                       nb_sampless=self.nb_samples, fd_etas=self.fd_eta,
                                                       estimator_type=self.zo_type,
                                                       query_until_success=self.query_until_succes)
        elif method == "FGSM":
            # print("Method of attack: {}".format(method))
            adversary = GradientSignAttack(
                self.msd.netV, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.26, targeted=False)
        elif method == "BIM":
            # print("Method of attack: {}".format(method))
            adversary = LinfBasicIterativeAttack(
                self.msd.netV, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
                nb_iter=200, eps_iter=0.02, clip_min=0.0, clip_max=1.0, targeted=False)
        elif method == "CW":
            # print("Method of attack: {}".format(method))
            adversary = CarliniWagnerL2Attack(
                self.msd.netV, num_classes=10, learning_rate=0.45, binary_search_steps=10,
                max_iterations=12, targeted=False)
        elif method == "PGD":
            # print("Method of attack: {}".format(method))
            adversary = PGDAttack(
                self.msd.netV, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.eps, eps_iter=self.eps_iter,
                nb_iter=self.nb_iter, clip_min=0.0, clip_max=1.0, targeted=False)
        elif method == "Linf-PGD":
            adversary = LinfPGDAttack(
                self.msd.netV, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.eps, eps_iter=self.eps_iter,
                nb_iter=self.nb_iter, clip_min=0.0, clip_max=1.0, targeted=False)
        elif method == "L2-PGD":
            # print("Method of attack: {}".format(method))
            adversary = L2PGDAttack(
                self.msd.netV, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.eps, eps_iter=self.eps_iter,
                nb_iter=self.nb_iter, clip_min=0.0, clip_max=1.0, targeted=False)
        elif method == "L1-PGD":
            adversary = L1PGDAttack(
                self.msd.netV, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.eps, eps_iter=self.eps_iter,
                nb_iter=self.nb_iter, clip_min=0.0, clip_max=1.0, targeted=False)
        elif method == "ZO-INF-PGD":
            adversary = NESAttack(self.msd.netV, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.eps,
                                  eps_iter=self.eps_iter,
                                  nb_iter=self.nb_iter, clip_min=0.0, clip_max=1.0, targeted=False)
        elif method == 'L2-MOM':
            adversary = L2MomentumIterativeAttack(
                self.msd.netV, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.eps, eps_iter=self.eps_iter,
                nb_iter=self.nb_iter, clip_min=0.0, clip_max=1.0, targeted=False)
        elif method == 'Linf-MOM':
            adversary = LinfMomentumIterativeAttack(
                self.msd.netV, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.eps, eps_iter=self.eps_iter,
                nb_iter=self.nb_iter, clip_min=0.0, clip_max=1.0, targeted=False)
        else:
            # Using clean data samples
            print("No Method for attack! Error")
            adversary = None
        # print(type(adversary).__name__)
        return adversary

    def create_adversary(self, method):
        self.adversary = self.get_adversary(method)

    def perturb(self, inputs, labels):
        if self.adversary is None:
            return inputs
        return self.adversary.perturb(inputs, labels)

    def attack(self):
        self.create_adversary(self.attack_name)

        correct = 0.0
        total = 0.0

        avg_queries = 0.0
        predicts = []
        iterations = 0.

        # make adversarial example
        for data in self.small_test_set:
            inputs, labels = data
            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()

            adv_inputs = self.perturb(inputs, labels)

            if self.query_until_succes and self.zo:
                avg_queries += self.adversary.get_avg_queries()
                iterations += 1.

            with torch.no_grad():
                outputs = self.msd.netV(adv_inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicts.append((predicted != labels).long().float())

                total += labels.size(0)
                correct += (predicted == labels).sum()

        if self.query_until_succes and self.zo:
            avg_queries = round((avg_queries / iterations).cpu().item(), 2)

        accuracy = round(1 - float(correct) / total, 3)
        variance = round((1 / (self.number_of_tests - 1)) * sum([torch.mul((p - accuracy), (p - accuracy)).sum().cpu().item() for p in predicts]), 3)
        # print("amount of tests: {}".format(total))
        # print("amount of correct: {}".format(correct))
        # print('Attack success rate of \'%s\': %.2f %%' % (self.attack_name, accuracy))
        # return adv_inputs, correct
        return accuracy, variance, avg_queries

    def example_image(self, index_images):
        self.create_adversary(self.attack_name)
        adv_images = []
        for index in index_images:
            example_image_org, example_label = self.test_set[index]

            # plt.imshow(self.example_image_org.numpy()[0], cmap='gray')
            # plt.savefig("plots/Example_images/{}_{}_original".format(self.dataset, self.example_label), bbox_inches='tight')
            # plt.show()

            example_img = torch.unsqueeze(example_image_org.cuda(), 1).cuda()
            example_label = torch.tensor(example_label).unsqueeze(-1).cuda()
            example_image_adv = self.perturb(example_img, example_label)
            example_output = self.msd.netV(example_image_adv)
            _, example_predicted = torch.max(example_output.data, 1)
            success = bool((example_predicted != example_label).sum())

            adv_images.append((example_image_org.numpy()[0], example_image_adv.cpu().numpy()[0][0], example_label, example_predicted.cpu().numpy()[0]))

            # plt.imshow(example_image_adv.cpu().numpy()[0][0], cmap='gray')
            # plt.axis('off')
            # plt.savefig("plots/Example_images/{}_{}_{}_adversarial_{}_{}".format(self.attack_name, self.dataset, self.zo_type,
            #                                                              example_predicted.cpu().numpy()[0], bool(success)), bbox_inches='tight')
            # plt.show()

        return adv_images

    @staticmethod
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data = torch.where(m.weight.data > 0, m.weight.data, torch.zeros(m.weight.data.shape))

    def get_number_of_queries(self):
        return self.number_of_queries

    # @staticmethod
    # def cross_entropy(q, p):
    #     return torch.mean(-torch.sum(p * torch.log(q), dim=1))
    #
    # def train_netS(self, path_s, path_g=None, data_type="REAL", label_only=False):
    #     if data_type == "REAL":
    #         self.train_netS_real(path_s, label_only)
    #     elif data_type == "synthetic":
    #         print("implement this according to the existing example")
    #     else:
    #         print("wrong data type")

    # def train_netS_real(self, path_s, label_only):
    #     """
    #     Training the substitute net using real samples to query.
    #     """
    #     print("Starting training net S using real samples to query.")
    #
    #     # optimizer_s = torch.optim.Adam(self.netS.parameters(), lr=self.args.lr)
    #     optimizer_s = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr)
    #     criterion = nn.CrossEntropyLoss()
    #
    #     self.netS.train()
    #     self.msd.netV.eval()
    #
    #     for epoch in range(self.args.epoch_g):
    #         print("epoch: %d/%d" % (epoch + 1, self.args.epoch_g))
    #
    #         for i, data in enumerate(self.train_loader, 0):
    #             # Updating netS
    #             self.netS.zero_grad()
    #
    #             x_query, _ = data
    #
    #             with torch.no_grad():
    #                 v_output = self.msd.netV(x_query)
    #                 v_output_p = F.softmax(v_output, dim=1)
    #                 _, v_predicted = torch.max(v_output_p, 1)
    #
    #             s_output = self.netS(x_query.detach())
    #             s_prob = F.softmax(s_output, dim=1)
    #
    #             if label_only:
    #                 loss_s = criterion(s_output, v_predicted)
    #             else:
    #                 loss_s = self.cross_entropy(s_prob, v_output_p)
    #
    #             loss_s.backward()
    #             optimizer_s.step()
    #
    #             if i % 200 == 0:
    #                 print("batch idx:", i, "loss_s:", loss_s.detach().numpy())
    #
    #     torch.save(self.netS.state_dict(), path_s)
    #     print("Finished training of netS")
