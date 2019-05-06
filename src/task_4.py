import copy
import random
import glob
import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt

class Vertex():
    def __init__(self, adj_list, D):
        self.adj = adj_list
        self.x = np.full(D, 1/D)


class GNN():
    def __init__(self, D, graph_path, label_path=None):
        with open(graph_path) as f:
            graph = f.readlines()

        if (label_path != None):
            with open(label_path) as f:
                self.label = int(f.read())
        else:
            self.label = -1

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

    def operation(self, W, A, b, T):
        for _ in range(T):
            self.conv_1()
            self.conv_2(W)
        L = self.calc_loss(A, b)
        return L

    def predict(self, A, b):
        h = self.readout()
        s = np.sum(np.dot(A, h)) + b
        pred = 1 if s > 0 else 0
        return pred, pred == self.label


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
        self.momentA_1 = np.zeros(self.D)
        self.momentA_2 = np.zeros(self.D)
        self.momentW_1 = np.zeros((self.D, self.D))
        self.momentW_2 = np.zeros((self.D, self.D))
        self.momentb_1 = 0
        self.momentb_2 = 0
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.adam_eps = 1e-8
        self.bias_boost = 1

    def optimizer_core(self, graph_list):
        num_graph = len(graph_list)
        gradA = np.zeros((num_graph, self.D))
        gradW = np.zeros((num_graph, self.D, self.D))
        gradb = np.zeros(num_graph)

        losses = []
        accs = []

        for g, graph in enumerate(graph_list):
            G_t = copy.deepcopy(graph)
            loss = graph.operation(self.W, self.A, self.b, self.T)
            _, acc = graph.predict(self.A, self.b)
            losses.append(loss)
            accs.append(acc)

            for i in range(self.D):
                tmpA = copy.deepcopy(self.A)
                tmpA[i] += self.eps
                G_tmp = copy.deepcopy(G_t)
                loss_tmp = G_tmp.operation(self.W, tmpA, self.b, self.T)
                gradA[g, i] = (loss_tmp - loss) / self.eps

            for i in range(self.D):
                for j in range(self.D):
                    tmpW = copy.deepcopy(self.W)
                    tmpW[i, j] += self.eps
                    G_tmp = copy.deepcopy(G_t)
                    loss_tmp = G_tmp.operation(self.W, self.A, self.b, self.T)
                    gradW[g, i, j] = (loss_tmp - loss) / self.eps 

            tmpb = copy.deepcopy(self.b)
            tmpb += self.eps
            G_tmp = copy.deepcopy(G_t)
            loss_tmp = G_tmp.operation(self.W, self.A, tmpb, self.T)
            gradb[g] = (loss_tmp - loss) / self.eps

        avg_gradb = np.mean(gradb)
        avg_gradA = np.mean(gradA, axis=0)
        avg_gradW = np.mean(gradW, axis=0)

        return losses, accs, avg_gradW, avg_gradA, avg_gradb

    def Adam(self, graph_list, timestep):
        losses, accs, avg_gradW, avg_gradA, avg_gradb = self.optimizer_core(graph_list)

        self.momentA_1 = self.beta_1 * self.momentA_1 + (1 - self.beta_1) * avg_gradA
        self.momentW_1 = self.beta_1 * self.momentW_1 + (1 - self.beta_1) * avg_gradW
        self.momentb_1 = self.beta_1 * self.momentb_1 + (1 - self.beta_1) * avg_gradb

        self.momentA_2 = self.beta_2 * self.momentA_2 + (1 - self.beta_2) * avg_gradA * avg_gradA
        self.momentW_2 = self.beta_2 * self.momentW_2 + (1 - self.beta_2) * avg_gradW * avg_gradW
        self.momentb_2 = self.beta_2 * self.momentb_2 + (1 - self.beta_2) * avg_gradb * avg_gradb

        bias_corrected_first_momentA = self.momentA_1 / (1 - self.beta_1 ** timestep)
        bias_corrected_first_momentW = self.momentW_1 / (1 - self.beta_1 ** timestep)
        bias_corrected_first_momentb = self.momentb_1 / (1 - self.beta_1 ** timestep)

        bias_corrected_second_momentA = self.momentA_2 / (1 - self.beta_2 ** timestep)
        bias_corrected_second_momentW = self.momentW_2 / (1 - self.beta_2 ** timestep)
        bias_corrected_second_momentb = self.momentb_2 / (1 - self.beta_2 ** timestep)

        self.A -= self.alpha * bias_corrected_first_momentA / (bias_corrected_second_momentA + self.adam_eps) ** 0.5
        self.W -= self.alpha * bias_corrected_first_momentW / (bias_corrected_second_momentW + self.adam_eps) ** 0.5
        self.b -= self.alpha * bias_corrected_first_momentb / (bias_corrected_second_momentb + self.adam_eps) ** 0.5

        return np.mean(losses), np.mean(accs)

    def validation(self, graph_list):
        pred = []
        losses = []
        for graph in graph_list:
            l = graph.operation(self.W, self.A, self.b, self.T)
            _, p = graph.predict(self.A, self.b)
            pred.append(p)
            losses.append(l)
        pred = np.array(pred)
        losses = np.array(losses)
        return np.mean(losses), np.sum(pred) / len(graph_list)
    
    def predict(self, graph):
        _ = graph.operation(self.W, self.A, self.b, self.T)
        p, _ = graph.predict(self.A, self.b)
        if p > 0.5:
            return 1
        else:
            return 0


class Trainer():
    def __init__(self, model_path=None):
        root_path = os.path.abspath(os.path.dirname(__file__))[:-3]
        graph_list = sorted(glob.glob(root_path + '/datasets/train/*_graph.txt'))
        label_list = sorted(glob.glob(root_path + '/datasets/train/*_label.txt'))
        data_full = list(zip(graph_list, label_list))
        data_length = len(data_full)
        self.dataset = data_full[:int(data_length * 0.8)]
        self.test_dataset = data_full[int(data_length*0.8):]
        self.data_num = len(self.dataset)

        self.batch_size = 32
        self.epochs = 50
        self.D = 8

        if (model_path == None):
            self.model = Classifier(self.D)
        else:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

    def train(self):
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        best_acc = 0
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
                    graphs.append(GNN(self.D, pair[0], pair[1]))
                loss, acc = self.model.Adam(graphs, iteration+1)
                self.epoch_loss += loss
                self.epoch_acc += acc

            print('epoch: ', epoch)
            train_loss = self.epoch_loss / self.batch_num
            train_acc = self.epoch_acc / self.batch_num
            print('train loss: ', train_loss)
            print('acc loss: ', train_acc)

            test_graphs = []
            for pair in self.test_dataset:
                test_graphs.append(GNN(self.D, pair[0], pair[1]))
            valid_loss, valid_acc = self.model.validation(test_graphs)
            print('valid loss', valid_loss)
            print('valid acc: ', valid_acc)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if (valid_acc > best_acc):
                print('Best valid acc updated! %f -> %f' % (best_acc, valid_acc))
                best_acc = valid_acc
                with open('model.pkl', mode='wb') as f:
                    pickle.dump(self.model, f)

        plt.plot(train_losses, label="train loss")
        plt.plot(valid_losses, label="valid loss")
        plt.title('Adam loss')
        plt.legend(loc="lower right")
        plt.savefig('Adam_loss.png')
        plt.show()

        plt.clf()

        plt.plot(train_accs, label="train acc")
        plt.plot(valid_accs, label="valid_acc")
        plt.title('Adam acc')
        plt.legend(loc="lower right")
        plt.savefig('Adam_acc.png')
        plt.show()

        plt.clf()

 
    
    def test(self):
        test_dataset_len = len(os.listdir('../datasets/test'))
        for i in range(test_dataset_len):
            basename = '%d_graph.txt' % i
            path = os.path.join('../datasets/test', basename)
            graph = GNN(self.D, path, None)
            p = self.model.predict(graph)
            with open('test_predict.txt', 'a') as f:
                f.write(str(p))
                f.write('\n')


if __name__ == "__main__":
    mode = sys.argv[1]

    if (mode == 'train'):
        trainer = Trainer(model_path=None)
        trainer.train()

    if (mode == 'test'):
        model_path = sys.argv[2]
        trainer = Trainer(model_path=model_path)
        trainer.test()


