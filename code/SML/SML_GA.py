import argparse
import datetime
import random
import time
from multiprocessing import Process, Lock, Manager

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


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


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


setup_seed(20)

parser = argparse.ArgumentParser(description="Test for argparse")
parser.add_argument('--processNum', '-p', type=int, default=2)
parser.add_argument('--epoch', '-e', type=int, default=6)
parser.add_argument('--populationSize', '-s', type=int, default=6)
parser.add_argument('--LSTM_epoch', '-L', type=int, default=100)
parser.add_argument('--layers', '-lay', type=int, default=1)
args = parser.parse_args()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.rnn = nn.LSTM(input_size, hidden_size, layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input):
        input = torch.transpose(input, 0, 1)
        input, _ = self.rnn(input)
        input = input[-1, :, :]
        input = self.fc(input)
        return input


def dataStander(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def dataInverse(data, scaler):
    mean = scaler.mean_[1]
    std = scaler.scale_[1]
    data = data * std + mean
    return data


def dataProcess(time_step, history=False, history_timestep=3):
    index = [i for i in range(2, 18)]
    index.extend([21, 22])
    df1 = pd.read_csv('../data/SML/NEW-DATA-1.csv', sep=',', usecols=index)
    # 3.14-4.11
    df1 = df1[0:]
    dataset1 = df1[:].values
    dataset1 = dataset1.astype('float32')

    df2 = pd.read_csv('../data/SML/NEW-DATA-2.csv', sep=',', usecols=index)
    # 4.18-5.2
    df2 = df2[0:]
    dataset2 = df2[:].values
    dataset2 = dataset2.astype('float32')
    dataset = np.concatenate((dataset1, dataset2), axis=0)
    # dataset, scaler = dataStander(dataset)
    train_set = dataset[:3200]

    train_set, scaler = dataStander(train_set)
    validate_set = dataset[3200:3600]
    validate_set = scaler.transform(validate_set)
    test_set = dataset[3600:]
    test_set = scaler.transform(test_set)
    train_x, train_y = dataSplit(train_set, time_step)
    validate_x, validate_y = dataSplit(validate_set, time_step)
    test_x, test_y = dataSplit(test_set, time_step)
    # return train_x, train_y, validate_x, validate_y, test_x, test_y
    return train_x, train_y, validate_x, validate_y, test_x, test_y, scaler


def dataSplit(dataset, timestep):
    data_x = [dataset[i:i + timestep] for i in range(len(dataset) - timestep)]
    data_x = np.array(data_x)
    data_y = [dataset[i + timestep, 1] for i in range(len(dataset) - timestep)]
    data_y = np.array(data_y)
    data_y = data_y.reshape([len(data_y), 1])
    return data_x, data_y


def GA(arrayIndividual, arrayFitness, id, loader, scaler, validate, result):
    validate_x = validate[0]
    validate_y = validate[1]
    populationSize = args.populationSize
    numGenerations = 5
    geneLength = 9
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    toolbox.register('binary', random.randint, 0, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=geneLength)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.1)
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
                receiveFitness = arrayFitness[(id + len(arrayIndividual) - 1) % len(arrayIndividual)]  # 取出适应度结果
                receiveIndividual = arrayIndividual[
                    (id + len(arrayIndividual) - 1) % len(arrayIndividual)]  # 取出该进程前面一个的迁移个体
                arrayIndividual[
                    (id + len(arrayIndividual) - 1) % len(arrayIndividual)] = []  # 将该位置list置为空，以方便上一个进程作为判断写入数据
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
        for jj in range(9):
            bb = bb + str(aa[0][jj])

        result.extend([int(bb, 2), aa[0].fitness.values[0]])
        print('进程{}迁移进化已经结束'.format(os.getpid()), int(bb, 2))
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
    for i in range(9):
        s = s + str(individual[i])
    hiddenSize = int(s[0:9], 2)

    input_size = 18
    layers = args.layers
    epoch = args.LSTM_epoch
    lr = 0.005
    rnn = RNN(input_size, hiddenSize, layers)
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
    #  rmse = mape(dataInverse(validate_temp.data.cpu().numpy(), scaler),
    #             dataInverse(validate_y_cuda.data.cpu().numpy(), scaler))
    #  rmse = r2_score(dataInverse(validate_temp.data.cpu().numpy(), scaler),
    #              dataInverse(validate_y_cuda.data.cpu().numpy(), scaler))
    rmse = np.sqrt(metrics.mean_squared_error(dataInverse(validate_temp.data.cpu().numpy(), scaler),
                                              dataInverse(validate_y_cuda.data.cpu().numpy(), scaler)))
    print('进程{}:个体{}的RMSE是{},值是{}'.format(os.getpid(), individual, rmse, hiddenSize))
    return rmse,


def main(args):
    startTime = datetime.datetime.now()
    batch_size = 100
    time_step = 10
    train_x, train_y, validate_x, validate_y, test_x, test_y, scaler = dataProcess(time_step)
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
    for i in range(processNum):
        p.apply_async(GA, args=(
            arrayIndividual, arrayFitness, i, loader, scaler, (validate_x, validate_y), result))  # cuda的数据不可以传送
    p.close()
    p.join()

    newIndividual = []
    newFitness = []
    for i in range(processNum):
        newIndividual.extend(arrayIndividual[i])
        newFitness.extend(arrayFitness[i])

    index = newFitness.index(min(newFitness))
    bestIndividual = newIndividual[index]

    s = ''
    for i in range(9):
        s = s + str(bestIndividual[i])

    bestIndividual = s
    hiddenSize = int(bestIndividual, 2)
    print(hiddenSize)
    endTime = datetime.datetime.now()
    print('时间是{}'.format(endTime - startTime))
    result = np.array(result).reshape(int(len(result) / 2), 2)
    data = pd.DataFrame(result, columns=['hiddenSize', 'rmse'])

    import os
    if not os.path.exists('../result/'):
        os.mkdir('../result/')

    path = '../result/SML2010_R2_processNum' + str(args.processNum) + '_populationSize' + str(
        args.populationSize) + '_LSTM_epoch' + str(args.LSTM_epoch) + '_epoch' + str(args.epoch) + '_0.005'
    writer = pd.ExcelWriter(path + '.xlsx')
    data.to_excel(writer, index=False)
    writer.save()
    writer.close()
    np.savetxt(path + '.txt', [hiddenSize, (endTime - startTime), newFitness[index]], delimiter=' ', fmt='%s')


if __name__ == "__main__":
    main(args)
