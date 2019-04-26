import copy
import random
import glob
import numpy as np
import sys


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
        return s, pred == self.label


def relu(v):
    vmax = np.vectorize(max)
    res = vmax(np.zeros_like(v), v)
    return res


class Classifier():
    def __init__(self, D):
        self.D = D
        self.eps= 0.001
        self.alpha = 0.01
        self.A = np.random.normal(0, 0.4, (self.D))
        self.W = np.random.normal(0, 0.4, (self.D, self.D))
        self.b = 0
        self.T = 1

    def optimizer(self, graph_list):
        num_graph = len(graph_list)
        gradA = np.zeros((num_graph, self.D))
        gradW = np.zeros((num_graph, self.D, self.D))
        gradb = np.zeros(num_graph)

        losses = []
        for g, graph in enumerate(graph_list):

            G_t = copy.deepcopy(graph)

            for t in range(self.T):
                graph.conv_1()
                graph.conv_2(self.W)
            loss = graph.calc_loss(self.A, self.b)
            losses.append(loss)

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

            tmpb = self.b
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

        self.A -= self.alpha * avg_gradA
        self.b -= self.alpha * avg_gradb
        self.W -= self.alpha * avg_gradW

        return np.mean(losses)

    def accuracy(self, graph_list):
        pred = []
        flag = []
        for graph in graph_list:
            for t in range(self.T):
                graph.conv_1()
                graph.conv_2(self.W)
            p, res = graph.predict(self.A, self.b)
            pred.append(p)
            flag.append(res)
        flag = np.array(flag)
        return np.sum(flag) / len(graph_list)


class Trainer():
    def __init__(self):
        graph_list = sorted(glob.glob('../datasets/train/*_graph.txt'))
        label_list = sorted(glob.glob('../datasets/train/*_label.txt'))
        data_full = list(zip(graph_list, label_list))
        data_length = len(data_full)
        self.dataset = data_full[:1000]
        self.test_dataset = data_full[1000:2000]
        self.data_num = len(self.dataset)

        self.batch_size = 8
        self.epochs = 100
        self.D = 8
        self.model = Classifier(self.D)

    def train(self):
        for epoch in range(self.epochs):
            random.shuffle(self.dataset)

            self.epoch_loss = 0
            self.graph_list = []
            self.batch_num = self.data_num // self.batch_size
            for iteration in range(self.batch_num):
                batch = self.dataset[iteration*self.batch_size:(iteration+1)*self.batch_size]
                graphs = []
                for pair in batch:
                    graphs.append(GNN(pair[0], pair[1], self.D))
                loss = self.model.optimizer(graphs)
                self.epoch_loss += loss

            print('epoch: ', epoch)
            print('loss: ', self.epoch_loss / self.batch_num)

            test_graphs = []
            for pair in self.test_dataset:
                test_graphs.append(GNN(pair[0], pair[1], self.D))
            acc = self.model.accuracy(test_graphs)
            print('acc: ', acc)

trainer = Trainer()
trainer.train()
