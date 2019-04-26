import math
import copy
import random

class Vertex():
    def __init__(self, adj_list, D):
        self.adj = adj_list
        self.x = [1/D] * D


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
            a_v = [0] * self.D
            for adj in self.V[i].adj:
                if adj == 1:
                    a_v = [x + y for (x, y) in zip(a_v, self.V[i].x)]
            self.a.append(a_v)

    def conv_2(self, W):
        for i, a_v in enumerate(self.a):
            self.V[i].x = relu(matMul(W, a_v))

    def readout(self):
        self.h = [0] * self.D
        for v in self.V:
            self.h = [x + y for (x, y) in zip(self.h, v.x)]

        return self.h

    def calc_loss(self, A, b):
        h = self.readout()
        s = sum([x * y for (x, y) in zip(A, h)]) + b
        L = self.label * math.log(1 + math.exp(-1 * s)) + (1 - self.label) * math.log(1 + math.exp(s))
        return L


def relu(v):
    res = []
    for e in v:
        res.append(max(0, e))
    return res

def matMul(W, a):
    res = []
    for row in W:
        prodsum = sum([x * y for (x, y) in zip(row, a)])
        res.append(prodsum)
    return res


class Classifier():
    def __init__(self, graph_path, label_path, D):
        self.D = D
        self.eps= 0.001
        self.alpha = 0.1
        self.GNN = GNN(graph_path, label_path, self.D)
        self.A = [random.gauss(0, 0.4) for i in range(self.D)]
        self.W = [[random.gauss(0, 0.4) for i in range(self.D)] for j in range(self.D)]
        self.b = 0
        self.T = 2

    def optimizer(self):
        newA = copy.deepcopy(self.A)
        newW = copy.deepcopy(self.W)
        newb = copy.deepcopy(self.b)

        G_t = copy.deepcopy(self.GNN)

        for t in range(self.T):
            self.GNN.conv_1()
            self.GNN.conv_2(self.W)
        loss = self.GNN.calc_loss(self.A, self.b)

        print(loss)

        for i in range(self.D):
            tmpA = copy.deepcopy(self.A)
            tmpA[i] += self.eps
            G_tmp = copy.deepcopy(G_t)
            for t in range(self.T):
                G_tmp.conv_1()
                G_tmp.conv_2(self.W)
            loss_tmp = G_tmp.calc_loss(tmpA, self.b)

            newA[i] -= self.alpha * (loss_tmp - loss) / self.eps

        for i in range(self.D):
            for j in range(self.D):
                tmpW = copy.deepcopy(self.W)
                tmpW[i][j] += self.eps
                G_tmp = copy.deepcopy(G_t)
                for t in range(self.T):
                    G_tmp.conv_1()
                    G_tmp.conv_2(tmpW)
                loss_tmp = G_tmp.calc_loss(self.A, self.b)

                newW[i][j] -= self.alpha * (loss_tmp - loss) / self.eps

        tmpb = self.b
        tmpb += self.eps
        G_tmp = copy.deepcopy(G_t)
        for t in range(self.T):
            G_tmp.conv_1()
            G_tmp.conv_2(self.W)
        loss_tmp = G_tmp.calc_loss(self.A, tmpb)

        newb -= self.alpha * (loss_tmp - loss) / self.eps

        self.A = newA
        self.W = newW
        self.b = newb


model = Classifier('../datasets/train/1_graph.txt', '../datasets/train/1_label.txt', 4)

for i in range(200):
    model.optimizer()

