import datetime
import random
import time
from multiprocessing import Process, Lock, Manager
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
import os
from sklearn.metrics import mean_squared_error, r2_score

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
import sklearn.metrics as metrics
import torch.utils.data as Data
from torch import tensor
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import warnings
import argparse
import time

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Test for argparse")
parser.add_argument('--processNum', '-p', type=int, default=2)
parser.add_argument('--epoch', '-e', type=int, default=50)
parser.add_argument('--populationSize', '-s', type=int, default=4)
parser.add_argument('--LSTM_epoch', '-L', type=int, default=100)
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


setup_seed(20)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def dataStander(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def dataInverse(data, scaler):
    min = scaler.data_min_[1]
    max = scaler.data_max_[1]
    data = data * (max - min) + min
    return data


def dataSplit(dataset, timestep):
    data_x = [dataset[i:i + timestep] for i in range(len(dataset) - timestep)]
    data_x = np.array(data_x)
    data_y = [dataset[i + timestep, 1] for i in range(len(dataset) - timestep)]
    data_y = np.array(data_y)
    data_y = data_y.reshape([len(data_y), 1])
    return data_x, data_y


def dataProcess(time_step):
    train_set = np.loadtxt('../data/land/train.txt')
    train_set = train_set.astype('float32')
    test_set = np.loadtxt('../data/land/test.txt')
    test_set = test_set.astype('float32')
    train_set, scaler = dataStander(train_set)
    test_set = scaler.transform(test_set)
    train_x, train_y = dataSplit(train_set, time_step)
    test_x, test_y = dataSplit(test_set, time_step)
    return train_x, train_y, test_x, test_y, scaler


class Rnn(nn.Module):
    def __init__(self, input_size, rnn_hidden_size1, hiddenSize2, num_layers):
        super(Rnn, self).__init__()
        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size1
        self.hiddenSize2 = hiddenSize2
        self.num_layers = num_layers
        self.sigmod = nn.Sigmoid()
        self.rnn1 = nn.LSTM(input_size, rnn_hidden_size1, num_layers)
        self.rnn2 = nn.LSTM(rnn_hidden_size1, hiddenSize2, num_layers)
        self.fc = nn.Linear(hiddenSize2, 1)

    def forward(self, input):
        input = torch.transpose(input, 0, 1)
        input, _ = self.rnn1(input)
        input, _ = self.rnn2(input)
        input = input[-1, :, :]
        input = self.fc(input)
        return input


def GA(arrayIndividual, arrayFitness, id, loader, scaler, validate, result):
    validate_x = validate[0]
    validate_y = validate[1]
    populationSize = args.populationSize
    numGenerations = 5
    geneLength = 18
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('binary', random.randint, 0, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=geneLength)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.1)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('evaluate', gaEvaluate, loader, validate_x, validate_y, scaler)

    population = toolbox.population(n=populationSize)
    for i in range(args.epoch):
        print('进程{}第{}次迭代进化开始'.format(os.getpid(), i + 1))
        r = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=numGenerations, verbose=False)
        print('进程{}第{}次迭代进化结束'.format(os.getpid(), i + 1))
        sendIndividual = tools.selBest(population, k=(populationSize // 2))
        sendFitness = [indi.fitness.values for indi in sendIndividual]
        while True:
            if len(arrayIndividual[id]):
                pass
            else:
                arrayFitness[id] = sendFitness
                t = []
                for n in range(len(sendIndividual)):
                    t.append(sendIndividual[n][:])
                arrayIndividual[id] = t
                print('进程{}写入的个体数据是{},适应度数据是{}'.format(os.getpid(), sendIndividual[:][:],
                                                       sendFitness))
                break

        while True:
            if len(arrayIndividual[(id + len(arrayIndividual) - 1) % len(arrayIndividual)]):
                receiveFitness = arrayFitness[(id + len(arrayIndividual) - 1) % len(arrayIndividual)]
                receiveIndividual = arrayIndividual[
                    (id + len(arrayIndividual) - 1) % len(arrayIndividual)]
                arrayIndividual[
                    (id + len(arrayIndividual) - 1) % len(arrayIndividual)] = []
                print('进程{}读出的个体数据是{}，适应度数据是{}'.format(os.getpid(), receiveIndividual[:][:],
                                                       receiveFitness))
                break

        receiveIndividual = [creator.Individual(j) for j in receiveIndividual]
        for ind, values in zip(receiveIndividual, receiveFitness):
            ind.fitness.values = values
        population.extend(receiveIndividual)
        best = tools.selBest(population, k=populationSize)
        population = best

        aa = tools.selBest(best, k=1)
        bb = ''
        for jj in range(18):
            bb = bb + str(aa[0][jj])
        cc1 = int(bb[0:9], 2)
        cc2 = int(bb[9:18], 2)
        result.extend([cc1, cc2, aa[0].fitness.values[0]])
        print('进程{}迁移进化已经结束'.format(os.getpid()))
        print('')

    bestIndividual = tools.selBest(population, k=1)
    while True:
        if len(arrayIndividual[id]):
            pass
        else:
            arrayFitness[id] = bestIndividual[0].fitness.values
            t = []
            t.append(bestIndividual[0][:])
            arrayIndividual[id] = t
            print('进程{}已经选出最佳的个体{},适应度数据为{}'.format(os.getpid(), bestIndividual[:][:],
                                                    arrayFitness[id]))
            break


def gaEvaluate(loader, validate_x, validate_y, scaler, individual):
    validate_x_cuda = Variable(torch.from_numpy(validate_x).cuda())
    validate_y_cuda = Variable(torch.from_numpy(validate_y).cuda())
    s = ''
    for i in range(18):
        s = s + str(individual[i])
    hiddenSize1 = int(s[0:9], 2)
    hiddenSize2 = int(s[9:18], 2)

    input_size = 2
    layers = 1
    epoch = args.LSTM_epoch
    lr = 0.005
    time_step = validate_x_cuda.size()[1]
    rnn = Rnn(input_size, hiddenSize1, hiddenSize2, layers)
    rnn = rnn.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(), lr, weight_decay=1e-5)
    for i in range(epoch):
        for step, (input, output) in enumerate(loader):
            rnn.train()
            input = Variable(input).cuda()
            output = Variable(output).cuda()
            out = rnn(input)
            loss = criterion(out, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    rnn.eval()
    validate_temp = rnn(validate_x_cuda)
    # rmse = r2_score(validate_y_cuda.data.cpu().numpy(), validate_temp.data.cpu().numpy())
    # rmse = mape(validate_temp.data.cpu().numpy(),validate_y_cuda.data.cpu().numpy())
    rmse = np.sqrt(metrics.mean_squared_error(dataInverse(validate_temp.data.cpu().numpy(), scaler),
                                              dataInverse(validate_y_cuda.data.cpu().numpy(), scaler)))
    # MAPE = np.mean(np.abs((validate_temp.data.cpu().numpy() - validate_y_cuda.data.cpu().numpy()) / validate_y_cuda.data.cpu().numpy()))
    # rmse = mape(dataInverse(validate_temp.data.cpu().numpy(), scaler),
    #             dataInverse(validate_y_cuda.data.cpu().numpy(), scaler))
    print('进程{}:个体{}的RMSE是{},值是{}和{}'.format(os.getpid(), individual, rmse, hiddenSize1, hiddenSize2))
    return rmse,

def main(args):
    startTime = datetime.datetime.now()
    batch_size = 100
    time_step = 10
    train_x, train_y, test_x, test_y, scaler = dataProcess(time_step)
    print(type(train_x),type(train_y),type(test_x),type(test_y))
    batch_dataset = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))  # 为了便于进行batch训练
    loader = Data.DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    arrayIndividual = Manager().list()
    arrayFitness = Manager().list()
    result = Manager().list()
    processNum = args.processNum
    for i in range(processNum):
        arrayIndividual.append([])
        arrayFitness.append([])
    p = Pool(processNum)
    print('into GA')
    for i in range(processNum):
        p.apply_async(GA, args=(
            arrayIndividual, arrayFitness, i, loader, scaler, (test_x, test_y), result))  # cuda的数据不可以传送
    p.close()
    p.join()
    print('out GA')
    newIndividual = []
    newFitness = []
    for i in range(processNum):
        newIndividual.extend(arrayIndividual[i])
        newFitness.extend(arrayFitness[i])

    index = newFitness.index(min(newFitness))
    bestIndividual = newIndividual[index]

    s = ''
    for i in range(18):
        s = s + str(bestIndividual[i])

    bestIndividual = s
    hiddenSize1 = int(bestIndividual[0:9], 2)
    hiddenSize2 = int(bestIndividual[9:18], 2)

    endTime = datetime.datetime.now()
    print('时间是{}'.format(endTime - startTime))

    result = np.array(result).reshape(int(len(result) / 3), 3)

    import os
    if not os.path.exists('../result/'):
        os.mkdir('../result/')

    path = '../result/land_MAPE_processNum' + str(args.processNum) + '_populationSize' + str(
        args.populationSize) + '_LSTM_epoch' + str(args.LSTM_epoch) + '_epoch' + str(args.epoch)
    writer = pd.ExcelWriter(path + '.xlsx')
    data = pd.DataFrame(result, columns=['hiddenSize1', 'hiddenSize1', 'rmse'])
    data.to_excel(writer, index=False)  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()

    np.savetxt('../result/land_MAPE_processNum' + str(args.processNum) + '_populationSize' + str(
        args.populationSize) + '_LSTM_epoch' + str(args.LSTM_epoch) + '_epoch' + str(args.epoch) + '.txt',
               [hiddenSize1, hiddenSize2, (endTime - startTime), newFitness[index]], delimiter=' ', fmt='%s')


if __name__ == "__main__":
    main(args)
