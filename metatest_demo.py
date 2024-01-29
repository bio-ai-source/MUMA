import os
import random
import copy
import torch
from torch import nn
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.functional import F
from net import Net, VNet
import argparse
from sklearn.metrics import confusion_matrix
import warnings

np.random.seed(0)


def seed_torch(seed=0):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--seed', type=int, default=1)

parser.set_defaults(augment=True)

args = parser.parse_args()
use_cuda = False
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

def build_model():
    model = Net()
    return model


class Lasso(nn.Module):
    def __init__(self, input_size):
        super(Lasso, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out


def soft_th(X, threshold=5):
    return np.sign(X) * np.maximum((np.abs(X) - threshold), np.zeros(X.shape))


def torch_soft_operator(X, threshold=0.1):
    np_X = X.numpy()
    tmp = np.sign(np_X) * np.maximum((np.abs(np_X) - threshold), np.zeros(np_X.shape))
    return torch.from_numpy(tmp.astype(np.float32))


def lasso(x, y, lmbda=1, lr=0.005, max_iter=2000, tol=1e-4, opt='SGD'):
    lso = Lasso(x.shape[1])
    criterion = nn.MSELoss(reduction='mean')
    if opt == 'adam':
        optimizer = optim.Adam(lso.parameters(), lr=lr)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(lso.parameters(), lr=lr)
    elif opt == "SGD":
        optimizer = optim.SGD(lso.parameters(), lr=lr)
    w_prev = torch.tensor(0.)
    for it in range(max_iter):
        # lso.linear.zero_grad()
        optimizer.zero_grad()
        out = lso(x)
        fn = criterion(out, y)
        l1_norm = lmbda * torch.norm(lso.linear.weight, p=1)
        l1_crit = nn.L1Loss()
        target = Variable(torch.from_numpy(np.zeros((x.shape[1], 1))))

        loss = 0.5 * fn + lmbda * F.l1_loss(lso.linear.weight, target=torch.zeros_like(lso.linear.weight.detach()),
                                            size_average=False)
        loss.backward()
        optimizer.step()
        # pdb.set_trace()
        if it == 0:
            w = lso.linear.weight.detach()
        else:
            with torch.no_grad():
                sign_w = torch.sign(w)
                ## hard-threshold
                lso.linear.weight[torch.where(torch.abs(lso.linear.weight) <= lmbda * lr)] = 0
                w = copy.deepcopy(lso.linear.weight.detach())
        if it % 500 == 0:
            #             print(target.shape)
            print(loss.item(), end=" ")
            print(torch.norm(lso.linear.weight.detach(), p=1), end=" ")
            print("l1_crit: ", l1_crit(lso.linear.weight.detach(), target), end=" ")
            print("F L1: ", F.l1_loss(lso.linear.weight.detach(), target=torch.zeros_like(lso.linear.weight.detach()),
                                      size_average=False))
    return lso.linear.weight.detach()


def synthetic_data(w, num_examples, avg, std):  # @save
    X = torch.normal(avg, std, (num_examples, len(w)))
    y = torch.matmul(X, w)
    y += torch.normal(0, 0.01, y.shape)
    y = torch.sigmoid(y)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    return X, y.reshape((-1, 1))


def norY(y):
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    return y


def sampleSelect(sample, x, y):
    pos = torch.where(sample == 3)
    neg = torch.where(sample == 0)
    X = x
    y_pos = y[pos]
    print(y_pos)
    y_neg = y[neg]
    y = torch.cat([y_pos, y_neg])
    print(y)
    # pos = pos.tolist(pos)
    pos_1 = [aa.tolist() for aa in pos]
    neg_1 = [aa.tolist() for aa in neg]
    pos_1 = pos_1[0]
    x_pos = x[pos_1, :]
    neg_1 = neg_1[0]
    x_neg = x[neg_1, :]
    x = torch.cat([x_pos, x_neg])
    return x, y


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield batch_indices


def sigmoid_net(X):
    return torch.sigmoid(torch.matmul(X, W))


class MultiLinearRegression(nn.Module):
    def __init__(self):
        super(MultiLinearRegression, self).__init__()
        self.linear = nn.Linear(featureDim, 1, bias=False)  

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)


class MultiLinearRegression1(nn.Module):
    def __init__(self, input, output):
        super(MultiLinearRegression1, self).__init__()
        self.linear = nn.Linear(input, output, bias=False) 

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)


class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


def sigmoid(x):
    x = x.detach().numpy()
    return (1. / (1. + np.exp(-x)))


def accuracy(y_hat, y):  # @save
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / len(y)


def synthetic_binary_confident_data(w, num_seeds, num_examples, avg, std, sigma):  # @save
    x = np.random.normal(avg, std, (num_seeds, len(w)))
    x = torch.tensor(x)
    y = torch.matmul(x.to(torch.float32), w)
    y += sigma*torch.normal(0, 0.5, y.shape)
    y = torch.sigmoid(y)
    y = np.array(y)
    indx = 0
    ratio_high = 90
    ratio_medium = 60
    indx_high = np.empty(0)
    indx_medium = np.empty(0)
    indx_low = np.empty(0)
    y_orgrin = np.array(y, copy=True)
    ans = np.argsort(y_orgrin)
    y_orgrin = y_orgrin[ans]
    x = x[ans, :]
    y_mark = np.zeros((len(y), 1))
    for value_y in y_orgrin:
        if value_y >= np.percentile(y_orgrin, ratio_high) or value_y <= np.percentile(y_orgrin, 100 - ratio_high):
            indx_high = np.append(indx_high, indx)
            y_mark[indx] = 1
            y[indx] = 1 if value_y >= 0.5 else 0
        elif value_y >= np.percentile(y_orgrin, ratio_medium) or value_y <= np.percentile(y_orgrin, 100 - ratio_medium):
            indx_medium = np.append(indx_medium, indx)
            y[indx] = 1 if value_y >= 0.5 else 0
            y_mark[indx] = 2
        else:
            indx_low = np.append(indx_low, indx)
            y[indx] = 1 if value_y >= 0.5 else 0
            y_mark[indx] = 3
        indx += 1
    ratio_high = int(num_examples*0.1)
    criterion = np.zeros(num_seeds, dtype=bool)
    criterion[0:ratio_high], criterion[-ratio_high:] = True, True
    criterion[ratio_high+100:ratio_high+100+ratio_high], criterion[-(ratio_high+100+ratio_high):-(ratio_high+100)] = True, True

    M = int(num_examples * 0.3)
    C = num_seeds // 2
    if y[C] == 1:
        criterion[C:C + M] = True
        criterion[C-150:C-150 + M] = True
    else:
        criterion[C - M:C] = True
        criterion[C+150:C+150 + M] = True

    x = x[criterion, :]
    y = y[criterion]
    y_orgrin = y_orgrin[criterion]
    y_mark = y_mark[criterion]
    return x.to(torch.float32), y, y_orgrin, y_mark


class Accumulator:  # @save

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):  # @save
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def sgd(params, lr, batch_size):  # @save
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.05


def updater(batch_size):
    return sgd([W], lr, batch_size)


def data_iter_meta(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def L2Loss(y, yhead):
    return torch.mean((y - yhead) ** 2)


def L1Loss(y, yhead):
    return torch.mean(torch.abs(y - yhead))


def train_auto(model, criterion, optimizer, x, y):
    yhat_auto = model(x).squeeze(-1)
    loss_auto = criterion(yhat_auto, y)
    loss_auto = loss_auto + lmbda * F.l1_loss(model.linear.weight,
                                              target=torch.zeros_like(model.linear.weight.detach()),
                                              size_average=False)
    optimizer.zero_grad()  #
    loss_auto.sum().backward()
    optimizer.step()


def meta_model_update(meta_model_1, meta_model_2,
                      vnet_1, vnet_2,
                      x_1, y_1,
                      x_2, y_2,
                      epoch):
    outputs = meta_model_1(x_1).squeeze(-1)
    cost = nn.BCELoss(reduction='none')
    cost = cost(outputs, y_1)  
    cost_v = torch.reshape(cost, (len(cost), 1))
    v_lambda = vnet_1(cost_v.data)
    w1 = abs((v_lambda - v_lambda.max()) / (v_lambda.max() - v_lambda.min()))

    tem, w2 = calculate_weight_loss2(meta_model_2, vnet_2, x_2, y_2)

    l_f_meta_1 = torch.sum(cost_v * w1) + 0.05 * F.l1_loss(meta_model_1.linear.weight,
                                                           target=torch.zeros_like(meta_model_1.linear.weight.detach()),
                                                           size_average=False) + 0.05 * L1Loss(w1, w2)
    meta_model_1.zero_grad()
    grads = torch.autograd.grad(l_f_meta_1, (meta_model_1.params()), create_graph=True)
    meta_lr = 1e-3
    meta_model_1.update_params(lr_inner=meta_lr, source_params=grads)
    del grads


def meta_model_update2(meta_model_1, meta_model_2, meta_model_3,
                       vnet_1, vnet_2, vnet_3,
                       x_1, y_1,
                       x_2, y_2,
                       x_3, y_3,
                       epoch):
    outputs = meta_model_1(x_1).squeeze(-1)
    outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
    cost = nn.BCELoss(reduction='none')
    cost = cost(outputs, y_1)  
    cost_v = torch.reshape(cost, (len(cost), 1))
    v_lambda = vnet_1(cost_v.data)
    w1 = abs((v_lambda - v_lambda.max()) / (v_lambda.max() - v_lambda.min()))

    tem, w2 = calculate_weight_loss2(meta_model_2, vnet_2, x_2, y_2)
    tem, w3 = calculate_weight_loss2(meta_model_3, vnet_3, x_3, y_3)

    l_f_meta_1 = torch.sum(cost_v * w1) + 0.05 * F.l1_loss(meta_model_1.linear.weight,
                                                           target=torch.zeros_like(meta_model_1.linear.weight.detach()),
                                                           size_average=False)  + 0.01*(L1Loss(w1,w2) + L1Loss(w1,w3) + L1Loss(w2,w3))/3
    meta_model_1.zero_grad()
    grads = torch.autograd.grad(l_f_meta_1, (meta_model_1.params()), create_graph=True)
    meta_lr = 1e-3
    meta_model_1.update_params(lr_inner=meta_lr, source_params=grads)
    del grads

def train_auto_meta3(model_1, model_2, model_3,
                     vnet_1, vnet_2, vnet_3,
                     optimizer_model_1, optimizer_model_2, optimizer_model_3,
                     optimizer_vnet_1, optimizer_vnet_2, optimizer_vnet_3,
                     x_1, y_1, x_2, y_2, x_3, y_3,
                     mx_1, my_1, mx_2, my_2, mx_3, my_3, epoch):
    model_1.train()
    my_1 = torch.tensor(my_1)
    y_1 = torch.tensor(y_1)

    model_2.train()  #
    my_2 = torch.tensor(my_2)
    y_2 = torch.tensor(y_2)
    model_3.train()
    my_3 = torch.tensor(my_3)
    y_3 = torch.tensor(y_3)
    ###########################################step 1#################################
    # meta_model_1 = MultiLinearRegression1(featureDim, 1)
    meta_model_1 = build_model()
    meta_model_1.load_state_dict(model_1.state_dict())

    meta_model_2 = build_model()
    meta_model_2.load_state_dict(model_2.state_dict())

    meta_model_3 = build_model()
    meta_model_3.load_state_dict(model_3.state_dict())

    meta_model_update2(meta_model_1, meta_model_2, meta_model_3,
                       vnet_1, vnet_2, vnet_3, x_1, y_1, x_2, y_2, x_3, y_3, epoch)
    #
    meta_model_update2(meta_model_2, meta_model_1, meta_model_3,
                       vnet_2, vnet_1, vnet_3, x_2, y_2, x_1, y_1, x_3, y_3, epoch)

    meta_model_update2(meta_model_3, meta_model_2, meta_model_1,
                       vnet_3, vnet_2, vnet_1, x_3, y_3, x_2, y_2, x_1, y_1, epoch)
    ###########################################step 2#################################
    mx_1 = mx_1.float()
    y_g_hat = meta_model_1(mx_1).squeeze(-1)
    y_g_hat = torch.where(torch.isnan(y_g_hat), torch.zeros_like(y_g_hat), y_g_hat)
    cost = nn.BCELoss(reduction='none')
    l_g_meta1 = cost(y_g_hat, my_1.to(torch.float32))
    l_g_meta1 = l_g_meta1 + lmbda* F.l1_loss(meta_model_1.linear.weight,
                                          target=torch.zeros_like(meta_model_1.linear.weight.detach()),
                                          size_average=True)  # + 0.01*(L1Loss(w1,w2) + L1Loss(w1,w3) + L1Loss(w2,w3))
    optimizer_vnet_1.zero_grad()
    l_g_meta1.sum().backward()
    optimizer_vnet_1.step()

    mx_2 = mx_2.float()
    y_g_hat2 = meta_model_2(mx_2).squeeze(-1)
    y_g_hat2 = torch.where(torch.isnan(y_g_hat2), torch.zeros_like(y_g_hat2), y_g_hat2)
    cost = nn.BCELoss(reduction='none')
    l_g_meta2 = cost(y_g_hat2, my_2.to(torch.float32))
    l_g_meta2 = l_g_meta2 + lmbda* F.l1_loss(meta_model_2.linear.weight,
                                          target=torch.zeros_like(meta_model_2.linear.weight.detach()),
                                          size_average=True)  # + 0.01*(L1Loss(w1,w2) + L1Loss(w1,w3) + L1Loss(w2,w3))
    optimizer_vnet_2.zero_grad()
    l_g_meta2.sum().backward()
    optimizer_vnet_2.step()

    mx_3 = mx_3.float()
    y_g_hat3 = meta_model_3(mx_3).squeeze(-1)
    y_g_hat3 = torch.where(torch.isnan(y_g_hat3), torch.zeros_like(y_g_hat3), y_g_hat3)
    cost = nn.BCELoss(reduction='none')
    l_g_meta3 = cost(y_g_hat3, my_3.to(torch.float32))

    l_g_meta3 = l_g_meta3 + lmbda* F.l1_loss(meta_model_3.linear.weight,
                                          target=torch.zeros_like(meta_model_3.linear.weight.detach()),
                                          size_average=True)  # + 0.01*(L1Loss(w1,w2) + L1Loss(w1,w3) + L1Loss(w2,w3))
    optimizer_vnet_3.zero_grad()
    l_g_meta3.sum().backward()
    optimizer_vnet_3.step()
    ###########################################step 3#################################
    loss_1, w1 = calculate_weight_loss2(model_1, vnet_1, x_1, y_1)
    loss_2, w2 = calculate_weight_loss2(model_2, vnet_2, x_2, y_2)
    loss_3, w3 = calculate_weight_loss2(model_3, vnet_3, x_3, y_3)

    loss_1 = loss_1 + lmbda * F.l1_loss(model_1.linear.weight,
                                       target=torch.zeros_like(model_1.linear.weight.detach()),
                                       size_average=False)  + lmbda2*(L1Loss(w1,w2) + L1Loss(w1,w3) + L1Loss(w2,w3))/3

    optimizer_model_1.zero_grad()
    loss_1.sum().backward()
    optimizer_model_1.step()

    loss_2 = loss_2 + lmbda * F.l1_loss(model_2.linear.weight,
                                       target=torch.zeros_like(model_2.linear.weight.detach()),
                                       size_average=False)  + lmbda2*(L1Loss(w1,w2) + L1Loss(w1,w3) + L1Loss(w2,w3))/3
    optimizer_model_2.zero_grad()
    loss_2.sum().backward()
    optimizer_model_2.step()

    loss_3 = loss_3 + lmbda * F.l1_loss(model_3.linear.weight,
                                       target=torch.zeros_like(model_3.linear.weight.detach()),
                                       size_average=False)  + lmbda2*(L1Loss(w1,w2) + L1Loss(w1,w3) + L1Loss(w2,w3))/3

    optimizer_model_3.zero_grad()
    loss_3.sum().backward()
    optimizer_model_3.step()
    #


def synthetic_meta_data(w, num_examples, num_metaExamples_ratio, avg, std):  # @save
    ratio = num_metaExamples_ratio
    x = np.random.normal(avg, std, (num_examples, len(w)))
    x = torch.tensor(x)
    y = np.matmul(x, w)
    y = sigmoid(y)
    y_orgrin = np.array(y, copy=True)
    ans = np.argsort(y_orgrin)
    y_orgrin = y_orgrin[ans]
    x = x[ans, :]

    Y1 = y_orgrin[0:int(len(y) * ratio)]
    Y1[:] = 0
    Y2 = y_orgrin[int(len(y) * (1 - ratio)):len(y)]
    Y2[:] = 1
    Y = np.concatenate((Y1, Y2), axis=0)
    # y =
    X1 = x[0:int(len(y) * ratio), :]
    X2 = x[int(len(y) * (1 - ratio)):len(y), :]
    X = np.concatenate((X1, X2), axis=0)
    X = torch.tensor(X)

    return X.to(torch.float32), Y


def calculate_weight_loss(model, vnet, x, y, flag):
    inputs, targets = x, y
    if flag == 0:
        outputs = model(inputs).squeeze(-1)
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        cost = nn.BCELoss(reduction='none')
        cost = cost(outputs, targets)
        cost_v = torch.reshape(cost, (len(cost), 1))

        with torch.no_grad():
            w_new = vnet(cost_v)
            w_new = abs((w_new - w_new.max()) / (w_new.max() - w_new.min()))


        return torch.sum(cost_v * w_new), w_new
    else:
        outputs = model(inputs).squeeze(-1)
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        cost = nn.BCELoss(reduction='none')
        cost = cost(outputs, targets)
        cost_v = torch.reshape(cost, (len(cost), 1))

        with torch.no_grad():
            w_new = vnet(cost_v)
            w_new = abs((w_new - w_new.max()) / (w_new.max() - w_new.min()))

        return torch.sum(cost_v * w_new), w_new


def calculate_weight_loss2(model, vnet, x, y):
    inputs, targets = x, y
    outputs = model(inputs).squeeze(-1)
    outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
    cost = nn.BCELoss(reduction='none')
    cost = cost(outputs, targets)  #
    cost_v = torch.reshape(cost, (len(cost), 1))

    with torch.no_grad():
        w_new = vnet(cost_v)
        w_new = abs((w_new - w_new.max()) / (w_new.max() - w_new.min()))

    return (cost_v * w_new), w_new
lmbda_list = []
lmbda2_list = []
acc_list = []
lmbda = 0.05
lmbda2 = 0.05
nums = 1
Sensitivity_M1 = 0
Specificity_M1 = 0
Sensitivity_M2 = 0
Specificity_M2 = 0
Sensitivity_M3 = 0
Specificity_M3 = 0
Sensitivity_all = 0
Specificity_all = 0
ACC_M = 0
ACC_M_voting = 0
ACC_M_handmake = 0
ACC_M_voting_handmake = 0
for num in range(nums):
    featureDim = 1000
    sampleNum = 200
    sigma = 0.01
    true_w_1 = torch.zeros(featureDim)
    true_w_1[0:8] = 1
    true_w_1[9:10] = -1
    true_w_2 = torch.zeros(featureDim)
    true_w_2[0:8] = 1
    true_w_2[9] = -1
    true_w_3 = torch.zeros(featureDim)
    true_w_3[0:8] = 1
    W = torch.normal(0, 0.01, size=(featureDim, 1), requires_grad=True)

    x1, y1, y1_origin, y1_mark = synthetic_binary_confident_data(true_w_1, 1000,sampleNum, 0, 1,sigma)
    x2, y2, y2_origin, y2_mark = synthetic_binary_confident_data(true_w_2, 1000,sampleNum, 0, 1,sigma)
    x3, y3, y3_origin, y3_mark = synthetic_binary_confident_data(true_w_3, 1000,sampleNum, 0, 1,sigma)

    x1_test, y1_test, y1_origin_test, y1_mark_test = synthetic_binary_confident_data(true_w_1, 1000,sampleNum, 0, 1,sigma)
    x2_test, y2_test, y2_origin_test, y2_mark_test = synthetic_binary_confident_data(true_w_2, 1000,sampleNum, 0, 1,sigma)
    x3_test, y3_test, y3_origin_test, y3_mark_test = synthetic_binary_confident_data(true_w_3, 1000,sampleNum, 0, 1,sigma)

    meta_data_size = int(sampleNum*0.1) + 1
    meta_x1 = np.concatenate((x1[0:meta_data_size], x1[sampleNum-meta_data_size:sampleNum]), axis=0)
    meta_x1 = torch.tensor(meta_x1)
    meta_y1 = np.concatenate((y1[0:meta_data_size], y1[sampleNum-meta_data_size:sampleNum]), axis=0)
    meta_x2 = np.concatenate((x2[0:meta_data_size], x2[sampleNum-meta_data_size:sampleNum]), axis=0)
    meta_x2 = torch.tensor(meta_x2)
    meta_y2 = np.concatenate((y2[0:meta_data_size], y2[sampleNum-meta_data_size:sampleNum]), axis=0)
    meta_x3 = np.concatenate((x3[0:meta_data_size], x3[sampleNum-meta_data_size:sampleNum]), axis=0)
    meta_x3 = torch.tensor(meta_x3)
    meta_y3 = np.concatenate((y3[0:meta_data_size], y3[sampleNum-meta_data_size:sampleNum]), axis=0)

    vnet_1 = VNet(1, 300, 1)
    model_1 = build_model()

    vnet_2 = VNet(1, 300, 1)
    model_2 = build_model()

    vnet_3 = VNet(1, 300, 1)
    model_3 = build_model()


    criterion_1 = nn.BCELoss(reduction='none')
    optimizer_1 = optim.SGD(model_1.params(), lr=1e-3)

    criterion_2 = nn.BCELoss(reduction='none')
    optimizer_2 = optim.SGD(model_2.params(), lr=1e-3)

    criterion_3 = nn.BCELoss(reduction='none')
    optimizer_3 = optim.SGD(model_3.params(), lr=1e-3)

    criterion_vnet_1 = nn.BCELoss(reduction='none')
    optimizer_vnet_1 = torch.optim.Adam(vnet_1.parameters(), 1e-2)

    criterion_vnet_2 = nn.BCELoss(reduction='none')
    optimizer_vnet_2 = torch.optim.Adam(vnet_2.parameters(), 1e-2)

    criterion_vnet_3 = nn.BCELoss(reduction='none')
    optimizer_vnet_3 = torch.optim.Adam(vnet_3.parameters(), 1e-2)

    loss = nn.BCELoss(reduction='none')

    num_epochs = 50
    batch_size = 10
    batch_size_meta = 10
    start = 0
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        for batch_indices in data_iter(batch_size, x1, y1):
            x1_subset = x1[batch_indices]
            y1_subset = y1[batch_indices]
            y1_subset = torch.FloatTensor(y1_subset).squeeze(-1)

            x1_subset_test = x1_test[batch_indices]
            y1_subset_test = y1_test[batch_indices]
            y1_subset_test = torch.FloatTensor(y1_subset_test).squeeze(-1)

            x2_subset = x2[batch_indices]
            y2_subset = y2[batch_indices]
            y2_subset = torch.FloatTensor(y2_subset).squeeze(-1)

            x2_subset_test = x2_test[batch_indices]
            y2_subset_test = y2_test[batch_indices]
            y2_subset_test = torch.FloatTensor(y2_subset_test).squeeze(-1)

            x3_subset = x3[batch_indices]
            y3_subset = y3[batch_indices]
            y3_subset = torch.FloatTensor(y3_subset).squeeze(-1)

            x3_subset_test = x3_test[batch_indices]
            y3_subset_test = y3_test[batch_indices]
            y3_subset_test = torch.FloatTensor(y3_subset_test).squeeze(-1)

            meta_x1_subset, meta_y1_subset = next(data_iter_meta(batch_size_meta, meta_x1, meta_y1))
            meta_x2_subset, meta_y2_subset = next(data_iter_meta(batch_size_meta, meta_x2, meta_y2))
            meta_x3_subset, meta_y3_subset = next(data_iter_meta(batch_size_meta, meta_x3, meta_y3))

            train_auto_meta3(model_1, model_2, model_3,
                             vnet_1, vnet_2, vnet_3,
                             optimizer_1, optimizer_2, optimizer_3,
                             optimizer_vnet_1, optimizer_vnet_2, optimizer_vnet_3,
                             x1_subset, y1_subset, x2_subset, y2_subset, x3_subset, y3_subset,
                             meta_x1_subset, meta_y1_subset, meta_x2_subset, meta_y2_subset, meta_x3_subset, meta_y3_subset,
                             epoch)

            ###
            yhat = sigmoid_net(x1_subset)
            yhat = yhat.squeeze(-1)
            l = loss(yhat, y1_subset) + lmbda * F.l1_loss(W, target=torch.zeros_like(W.detach()),
                                                          size_average=False)

            l.sum().backward()
            updater(x1_subset.shape[0])

            if start == 0:
                start = 1
            else:
                with torch.no_grad():
                    sign_w = torch.sign(W)
                    model_1.linear.weight[torch.where(torch.abs(model_1.linear.weight) <= lmbda * lr)] = 0
                    model_2.linear.weight[torch.where(torch.abs(model_2.linear.weight) <= lmbda * lr)] = 0
                    model_3.linear.weight[torch.where(torch.abs(model_3.linear.weight) <= lmbda * lr)] = 0


