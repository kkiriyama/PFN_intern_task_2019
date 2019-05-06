import copy
import random
import glob
import numpy as np
import sys
import os
import matplotlib.pyplot as plt


class Vertex():
    def __init__(self, adj_list, D):
        self.adj = adj_list
        self.x = np.full(D, 1/D)


class GNN():
    def __init__(self, graph_path, label_path, D):
        with open(graph_path) as f:
            graph = f.readlines()
        with open(label_path) as f:
            label = f.read()

        self.label = int(label)
        self.size = int(graph[0])
        self.graph = [[int(element.strip()) for element in row.split(' ')] for row in graph[1:]]

        self.D = D

        self.V = []
        for i in range(self.size):
            self.V.append(Vertex(self.graph[i], self.D))

    def conv_1(self):
        self.a = []
        for i in range(self.size):
            a_v = np.sum([self.V[j].x for j in self.V[i].adj if j == 1])
            self.a.append(a_v)

    def conv_2(self, W):
        for i, a_v in enumerate(self.a):
            self.V[i].x = relu(np.dot(W, a_v))

    def readout(self):
        self.h = np.sum([v.x for v in self.V], axis=0)
        return self.h

    def calc_loss(self, A, b):
        h = self.readout()
        s = np.sum(np.dot(A, h)) + b
        if s > 34:
            L = self.label * 1e-15 + (1 - self.label) * s
            return L
        if s < -34:
            L = self.label * abs(s) + (1-self.label) * 1e-15
            return L
        L = self.label * np.log(1 + np.exp(-1 * s)) + (1 - self.label) * np.log(1 + np.exp(s))
        return L

    def predict(self, A, b):
        h = self.readout()
        s = np.sum(np.dot(A, h)) + b
        pred = 1 if s > 0 else 0
        return pred == self.label


def relu(v):
    vmax = np.vectorize(max)
    res = vmax(np.zeros_like(v), v)
    return res


class Classifier():
    def __init__(self, D):
        self.D = D
        self.eps= 0.001
        self.alpha = 0.0001
        self.A = np.random.normal(0, 0.4, (self.D))
        self.W = np.random.normal(0, 0.4, (self.D, self.D))
        self.b = 0
        self.T = 1
        self.nu = 0.9
        self.momentA = np.zeros(self.D)
        self.momentW = np.zeros((self.D, self.D))
        self.momentb = 0
        self.bias_boost = 100

    def SGDcore(self, graph_list):
        num_graph = len(graph_list)
        gradA = np.zeros((num_graph, self.D))
        gradW = np.zeros((num_graph, self.D, self.D))
        gradb = np.zeros(num_graph)

        losses = []
        accs = []
        for g, graph in enumerate(graph_list):

            G_t = copy.deepcopy(graph)

            for t in range(self.T):
                graph.conv_1()
                graph.conv_2(self.W)
            loss = graph.calc_loss(self.A, self.b)
            acc = graph.predict(self.A, self.b)
            losses.append(loss)
            accs.append(acc)

            for i in range(self.D):
                tmpA = copy.deepcopy(self.A)
                tmpA[i] += self.eps
                G_tmp = copy.deepcopy(G_t)

                for t in range(self.T):
                    G_tmp.conv_1()
                    G_tmp.conv_2(self.W)
                loss_tmp = G_tmp.calc_loss(tmpA, self.b)
                gradA[g, i] = (loss_tmp - loss) / self.eps

            for i in range(self.D):
                for j in range(self.D):
                    tmpW = copy.deepcopy(self.W)
                    tmpW[i, j] += self.eps
                    G_tmp = copy.deepcopy(G_t)
                    for t in range(self.T):
                        G_tmp.conv_1()
                        G_tmp.conv_2(tmpW)
                    loss_tmp = G_tmp.calc_loss(self.A, self.b)

                    gradW[g, i, j] = (loss_tmp - loss) / self.eps 

            tmpb = copy.deepcopy(self.b)
            tmpb += self.eps
            G_tmp = copy.deepcopy(G_t)
            for t in range(self.T):
                G_tmp.conv_1()
                G_tmp.conv_2(self.W)
            loss_tmp = G_tmp.calc_loss(self.A, tmpb)

            gradb[g] = (loss_tmp - loss) / self.eps

        avg_gradb = np.mean(gradb)
        avg_gradA = np.mean(gradA, axis=0)
        avg_gradW = np.mean(gradW, axis=0)

        return losses, accs, avg_gradW, avg_gradA, avg_gradb

    def SGD(self, graph_list):
        losses, accs, avg_gradW, avg_gradA, avg_gradb = self.SGDcore(graph_list)

        self.A -= self.alpha * avg_gradA
        self.b -= self.alpha * avg_gradb * self.bias_boost
        self.W -= self.alpha * avg_gradW

        return np.mean(losses), np.mean(accs)

    def validation(self, graph_list):
        flag = []
        losses = []
        for graph in graph_list:
            for t in range(self.T):
                graph.conv_1()
                graph.conv_2(self.W)
            is_correct = graph.predict(self.A, self.b)
            loss = graph.calc_loss(self.A, self.b)
            flag.append(is_correct)
            losses.append(loss)
        flag = np.array(flag)
        loss = np.array(loss)
        return np.mean(loss), np.sum(flag) / len(graph_list)

    def momentumSGD(self, graph_list):
        losses, accs, avg_gradW, avg_gradA, avg_gradb = self.SGDcore(graph_list)
        self.A = self.A - self.alpha * avg_gradA + self.nu * self.momentA
        self.b = self.b - self.alpha * avg_gradb * self.bias_boost + self.nu * self.momentb
        self.W = self.W - self.alpha * avg_gradW + self.nu * self.momentW

        self.momentA = self.alpha * avg_gradA
        self.momentb = self.alpha * avg_gradb * self.bias_boost
        self.momentW = self.alpha * avg_gradW

        return np.mean(losses), np.mean(accs)


class Trainer():
    def __init__(self):
        root_path = os.path.abspath(os.path.dirname(__file__))[:-3]
        graph_list = sorted(glob.glob(root_path + '/datasets/train/*_graph.txt'))
        label_list = sorted(glob.glob(root_path + '/datasets/train/*_label.txt'))
        data_full = list(zip(graph_list, label_list))
        data_length = len(data_full)
        self.dataset = data_full[:int(data_length * 0.8)]
        self.test_dataset = data_full[int(data_length * 0.8):]
        self.data_num = len(self.dataset)

        self.batch_size = 8
        self.epochs = 20
        self.D = 4
        self.model = Classifier(self.D)

    def train(self):
        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = []
        for epoch in range(self.epochs):
            random.shuffle(self.dataset)

            self.epoch_loss = 0
            self.epoch_acc = 0
            self.graph_list = []
            self.batch_num = self.data_num // self.batch_size
            for iteration in range(self.batch_num):
                batch = self.dataset[iteration*self.batch_size:(iteration+1)*self.batch_size]
                graphs = []
                for pair in batch:
                    graphs.append(GNN(pair[0], pair[1], self.D))
                if(sys.argv[1] == 'SGD'):
                    loss, acc = self.model.SGD(graphs)
                elif(sys.argv[1] == 'momentumSGD'):
                    loss, acc = self.model.momentumSGD(graphs)
                else:
                    print('Unknown optimizer name')
                self.epoch_loss += loss
                self.epoch_acc += acc
            train_loss = self.epoch_loss / self.batch_num
            train_acc = self.epoch_acc / self.batch_num

            test_graphs = []
            for pair in self.test_dataset:
                test_graphs.append(GNN(pair[0], pair[1], self.D))
            valid_loss, valid_acc = self.model.validation(test_graphs)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            print('train loss: ', train_loss)
            print('train acc: ', train_acc)
            print('valid_loss', valid_loss)
            print('valid_acc', valid_acc)

        plt.plot(train_losses, label='train loss')
        plt.plot(valid_accs, label='valid loss')
        plt.title('momentumSGD loss')
        plt.legend(loc='lower_right')
        plt.savefig('momentumSGD_loss.png')
        plt.show()

        plt.clf()

        plt.plot(train_accs, label='train acc')
        plt.plot(valid_accs, label='valid acc')
        plt.title('momentumSGD acc')
        plt.legend(loc='lower_right')
        plt.savefig('momentumSGD_acc.png')
        plt.show()

        plt.clf()

trainer = Trainer()
trainer.train()

