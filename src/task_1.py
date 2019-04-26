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
            a_v = [0] * self.D
            for adj in self.V[i].adj:
                if adj == 1:
                    a_v = [x + y for (x, y) in zip(a_v, self.V[i].x)]
            self.a.append(a_v)

    def conv_2(self):
        self.W = [[1/self.D**2] * self.D] * self.D
        for i, a_v in enumerate(self.a):
            self.V[i].x = relu(matMul(self.W, a_v))

    def readout(self):
        self.h = [0] * self.D
        for v in self.V:
            self.h = [x + y for (x, y) in zip(self.h, v.x)]

        return self.h

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

gnn = GNN('../datasets/train/0_graph.txt', 4)
for i in range(1000):
    gnn.conv_1()
    gnn.conv_2()
h = gnn.readout()
print(h)
