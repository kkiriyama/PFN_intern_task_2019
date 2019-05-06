import numpy as np

class Vertex():
    def __init__(self, adj_list, D):
        self.adj = adj_list
        self.x = [1/D] * D

class GNN():
    def __init__(self, graph_path, D):
        with open(graph_path) as f:
            graph = f.readlines()

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

    def conv_2(self):
        self.W = [[1/self.D**2] * self.D] * self.D
        for i, a_v in enumerate(self.a):
            self.V[i].x = relu(np.dot(self.W, a_v))

    def readout(self):
        self.h = np.sum([v.x for v in self.V], axis=0)
        return self.h

def relu(v):
    vmax = np.vectorize(max)
    res = vmax(np.zeros_like(v), v)
    return res

gnn = GNN('../datasets/train/0_graph.txt', 4)
for i in range(2):
    gnn.conv_1()
    gnn.conv_2()
h = gnn.readout()
print(h)
