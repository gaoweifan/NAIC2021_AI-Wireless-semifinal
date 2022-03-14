from __future__ import division
import numpy as np
# import scipy.interpolate 
import math
import os
from multiprocessing import Process,Pool,Manager
import time

mu = 2
K = 256
CP = 32

mode=0
SNRdb=10
Pilotnum=8

def print_something():
    print('utils.py has been loaded perfectly')


def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL * sigma
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx] * CL), abs(x_clipped[clipped_idx]))
    return x_clipped


def PAPR(x):
    Power = np.abs(x) ** 2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10 * np.log10(PeakP / AvgP)
    return PAPR_dB


def Modulation(bits, mu):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return 0.7071 * (2 * bit_r[:, 0] - 1) + 0.7071j * (2 * bit_r[:, 1] - 1)  # This is just for QAM modulation


def deModulation(Q):
    Qr=np.real(Q)
    Qi=np.imag(Q)
    bits=np.zeros([64,2])
    bits[:,0]=Qr>0
    bits[:,1]=Qi>0
    return bits.reshape([-1])  # This is just for QAM modulation

def Modulation1(bits, mu):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return (bit_r[:, 0]) + 1j * (bit_r[:, 1])


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time, CP, CP_flag, mu, K):
    if CP_flag == False:
        # add noise CP
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        codeword_noise = Modulation(bits_noise, mu)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]  # take the last CP samples ...
    # cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse, SNRdb):

    convolved = np.convolve(signal, channelResponse)

    sigma2 = 0.0015 * 10 ** (-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise


def removeCP(signal, CP, K):
    return signal[CP:(CP + K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))



def ofdm_simulate(codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,
                  Clipping_Flag):

    # --- training inputs ----

    CR=1
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers

    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, 2 * K)
    # OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX, CR)  # add clipping
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP,  K)
    OFDM_RX_noCP = np.fft.fft(OFDM_RX_noCP)
    # OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword, mu)
    if len(codeword_qam) != K:
        print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    # OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword, CR)  # add clipping
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword, CP,  K)
    OFDM_RX_noCP_codeword = np.fft.fft(OFDM_RX_noCP_codeword)
    AA = np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP)))


    CC=OFDM_RX_noCP/np.max(AA)
    BB = np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword)))

    return np.concatenate((AA, BB)), CC  # sparse_mask


def MIMO(X, HMIMO, SNRdb,flag,P):
    P = P * 2
    Pilot_file_name = 'Pilot_' + str(P)
    if os.path.isfile(Pilot_file_name):
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(P * mu,))
        np.savetxt(Pilot_file_name, bits, delimiter=',')
    pilotValue = Modulation(bits, mu)


    if flag==1:
        cpflag, CR = 0, 0
    elif flag==2:
        cpflag, CR = 0, 1
    else:
        cpflag, CR = 1, 0
    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // P)
    dataCarriers = [val for val in allCarriers if not (val in pilotCarriers)]



    bits0=X[0]
    bits1=X[1]
    pilotCarriers1 = pilotCarriers[0:P:2]
    pilotCarriers2 = pilotCarriers[1:P:2]
    signal_output00, para = ofdm_simulate(bits0, HMIMO[0,:], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:P:2],
                                                    pilotCarriers1, dataCarriers, CR)
    signal_output01, para = ofdm_simulate(bits0, HMIMO[1, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:P:2],
                                          pilotCarriers1, dataCarriers, CR)
    signal_output10, para = ofdm_simulate(bits1, HMIMO[2, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:P:2],
                                          pilotCarriers2, dataCarriers, CR)
    signal_output11, para = ofdm_simulate(bits1, HMIMO[3, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:P:2],
                                          pilotCarriers2, dataCarriers, CR)

    signal_output0=signal_output00+signal_output10
    signal_output1=signal_output01+signal_output11
    output=np.concatenate((signal_output0, signal_output1))
    output=np.transpose(np.reshape(output,[8,-1]),[1,0])

    #print(np.shape(signal_output00))
    return np.reshape(output,[-1])


def MIMO4x16(X, HMIMO, SNRdb,flag,P):
    P = P * 4
    Pilot_file_name = 'Pilot_' + str(P)
    if os.path.isfile(Pilot_file_name):
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(P * mu,))
        np.savetxt(Pilot_file_name, bits, delimiter=',')
    pilotValue = Modulation(bits, mu)


    if flag==1:
        cpflag, CR = 0, 0
    elif flag==2:
        cpflag, CR = 0, 1
    else:
        cpflag, CR = 1, 0
    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // P)
    dataCarriers = [val for val in allCarriers if not (val in pilotCarriers)]



    bits0 = X[0]
    bits1 = X[1]
    bits2 = X[2]
    bits3 = X[3]
    pilotCarriers1 = pilotCarriers[0:P:4]
    pilotCarriers2 = pilotCarriers[1:P:4]
    pilotCarriers3 = pilotCarriers[2:P:4]
    pilotCarriers4 = pilotCarriers[3:P:4]
    real_sinal_output=[]
    for i in range(16):


        signal_output00, para = ofdm_simulate(bits0, HMIMO[i*4,:], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:P:4],
                                                    pilotCarriers1, dataCarriers, CR)

        signal_output10, para = ofdm_simulate(bits1, HMIMO[i*4+1, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:P:4],
                                          pilotCarriers2, dataCarriers, CR)

        signal_output20, para = ofdm_simulate(bits2, HMIMO[i*4+2, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[2:P:4],
                                          pilotCarriers3, dataCarriers, CR)

        signal_output30, para = ofdm_simulate(bits3, HMIMO[i*4+3, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[3:P:4],
                                          pilotCarriers4, dataCarriers, CR)
        real_sinal_output.append(signal_output00+signal_output10+signal_output20+signal_output30)
    output=np.asarray(real_sinal_output)
    output=np.transpose(np.reshape(output,[16*2*2,-1]),[1,0])
    return np.reshape(output,[-1])



####################使用链路和信道数据产生训练数据##########

################产生模拟验证环境########
def generatorXY(batch, H,h_idx=0):
    print("generating validation set")
    start_time = time.time()

    input_labels = []
    input_samples = []
    for _ in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits2 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits3 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1, bits2, bits3]
        # h_idx = np.random.randint(0, len(H))
        h_idx=h_idx%9000
        HH = H[h_idx]
        h_idx=h_idx+1
        YY = MIMO4x16(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1, bits2, bits3), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = np.asarray(input_samples,dtype=np.float32)
    batch_x = np.asarray(input_labels)
    #print(np.shape(batch_y))# (1000, 16384)
    #print(np.shape(batch_x))# (1000, 2048)

    end_time = time.time()
    print(' Took %f seconds to generate %d samples' % (end_time - start_time,batch))

    return (batch_y, batch_x)

def generator(batch,H,h_idx=0):
    while True:
        # start_time = time.time()

        input_labels = []
        input_samples = []
        for _ in range(0, batch):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits2 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits3 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            X = [bits0, bits1, bits2, bits3]
            # h_idx = np.random.randint(0, len(H))
            h_idx=h_idx%9000
            HH = H[h_idx]
            h_idx=h_idx+1
            YY = MIMO4x16(X, HH, SNRdb, mode, Pilotnum) / 20  ###
            XX = np.concatenate((bits0, bits1, bits2, bits3), 0)
            input_labels.append(XX)
            input_samples.append(YY)
        batch_y = np.asarray(input_samples,dtype=np.float32)
        batch_x = np.asarray(input_labels)
        #print(np.shape(batch_y))# (1000, 16384)
        #print(np.shape(batch_x))# (1000, 2048)

        # end_time = time.time()
        # print(' Took %f seconds to generate %d samples' % (end_time - start_time,batch))

        yield (batch_y, batch_x)

def generator_mp(batch,H,queue,h_idx=0):
    input_labels = []
    input_samples = []
    for _ in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits2 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits3 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1, bits2, bits3]
        # h_idx = np.random.randint(0, len(H))
        h_idx=h_idx%9000
        HH = H[h_idx]
        h_idx=h_idx+1
        YY = MIMO4x16(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1, bits2, bits3), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = np.asarray(input_samples,dtype=np.float32)
    batch_x = np.asarray(input_labels)
    queue.put((batch_y, batch_x))
    # dataSet.append((batch_y, batch_x))

def multiGen(each_batch,H,worker,epoch,queue):
    pool = Pool(worker)
    for _ in range(epoch):
        start_time = time.time()
        pool.starmap(generator_mp, [(each_batch,H,queue,each_batch*i) for i in range(0,worker)])
        end_time = time.time()
        print('\nTook %f seconds to generate all %d samples\n' % (end_time - start_time,each_batch*worker))
    pool.close()
    pool.join()

# 每次迭代返回一个批次的样本
# 每批次样本数=each_batch/batch_divider
# 每轮样本数=each_batch*worker(其中worker=steps_per_epoch)
# 总样本数=each_batch*worker*epoch
def multiGenerator(H,epoch,each_batch=1000,batch_divider=1,worker=10):
    queue = Manager().Queue()
    getter_process = Process(target=multiGen, args=(each_batch,H,worker,epoch,queue))
    getter_process.start()
    i=0
    while True:
        i=i+1
        if(i>worker*epoch):
            batch_y, batch_x = generatorXY(each_batch, H)
        else:
            dataSet=queue.get(True)
            batch_y=dataSet[0]
            batch_x=dataSet[1]
        devided_batch_ys = np.split(batch_y,batch_divider,0)
        devided_batch_xs = np.split(batch_x, batch_divider, 0)
        for i,devided_batch_y in enumerate(devided_batch_ys):
            yield (devided_batch_y, devided_batch_xs[i])
# 每次迭代返回一个批次的样本
# 需读取文件数=Nf
# 总样本数    =9000*Nf
# 每轮样本数  =9000*Ne
# 每批次样本数=9000*Ne/steps_per_epoch
def offLineGenerator(H,Nf,Ne,steps_per_epoch):
    data_load_address = './data'
    i=0
    while True:
        i=i+1
        if(i>Nf):
            batch_y, batch_x = generatorXY(int(9000*Ne/steps_per_epoch), H)
            batch_ys = [batch_y]
            batch_xs = [batch_x]
        else:
            print("\nreading data from file")
            start_time = time.time()
            dataSetName = os.listdir(data_load_address+'/trainSet/')
            while(len(dataSetName)<=1):
                print("data not generated yet")
                time.sleep(5)
                dataSetName = os.listdir(data_load_address+'/trainSet/')
            fileIdx=1
            uuidStr=dataSetName[fileIdx].split(".")[0]
            while(dataSetName.count(uuidStr)==0):
                fileIdx=(fileIdx+1)%len(dataSetName)
                if(fileIdx==0):
                    print(uuidStr,"data not ready")
                    time.sleep(0.5)
                    fileIdx=1
                dataSetName = os.listdir(data_load_address+'/trainSet/')
                uuidStr=dataSetName[fileIdx].split(".")[0]
            print(uuidStr+'.npy')
            all=np.load(data_load_address+'/trainSet/'+uuidStr+'.npy').astype(np.float32)
            os.remove(data_load_address+'/trainSet/'+uuidStr)
            os.remove(data_load_address+'/trainSet/'+uuidStr+'.npy')
            batch_x=all[:2048].T
            batch_y=all[2048:].T
            end_time = time.time()
            print('\nTook %f seconds to read 9000 training data\n' % (end_time - start_time))
            batch_divider = int(steps_per_epoch/Ne)#9000/batch_size
            batch_ys = np.split(batch_y, batch_divider, 0)
            batch_xs = np.split(batch_x, batch_divider, 0)
            for _ in range(5):
                batch_ys.extend(batch_ys)
                batch_xs.extend(batch_xs)
        for i,devided_batch_y in enumerate(batch_ys):
            yield (devided_batch_y, batch_xs[i])


#评价指标
def score_train(y_true, y_pred):
    import tensorflow as tf
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_pred = tf.cast(tf.floor(y_pred + 0.5), dtype=tf.int32)
    same=tf.cast(y_true==y_pred, dtype=tf.float32)
    score=100*tf.reduce_mean(same)
    return score
    # y_pred = np.array(np.floor(y_pred + 0.5), dtype=bool)
    # return 100*np.mean(y_true==y_pred)

import mindspore as ms
from mindspore import nn
from mindspore.train.callback import Callback
class scoreMAE(nn.Metric):
    """定义metric"""
    def __init__(self):
        super(scoreMAE, self).__init__()
        self.clear()

    def clear(self):
        self.abs_error_sum = 0
        self.samples_num = 0

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy()
        y_pred = np.floor(y_pred + 0.5)
        y = inputs[1].asnumpy()
        y = np.floor(y + 0.5)
        error_abs = np.abs(y.reshape(y_pred.shape) - y_pred)
        self.abs_error_sum += error_abs.sum()
        self.samples_num += y.shape[0]

    def eval(self):
        return self.abs_error_sum / self.samples_num

class SaveCallback(Callback):
    def __init__(self, eval_model, ds_eval,filepath, per_print_times=1):
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.acc = 0
        self.train_network = None
        self.filepath=filepath
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.start_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            print('%f seconds per epoch' % (time.time() - self.start_time))
            
            # print(cb_params)
            # print(cb_params.train_dataset)
            # print(cb_params.epoch_num)
            # print(cb_params.batch_num)
            # print(cb_params.train_network)
            # print(cb_params.cur_epoch_num)
            # print(cb_params.cur_step_num)
            # print(cb_params.parallel_mode)
            # print(cb_params.net_outputs)
            start_time = time.time()
            result = self.model.eval(self.ds_eval)
            end_time = time.time()
            print('Took %f seconds to evaluate' % (end_time - start_time))
            print("score:",result['score'])
            if result['score'] > self.acc:
                print(f"best score update from {self.acc} to {result['score']}")
                self.acc = result['score']
                self.train_network = cb_params.train_network
                # file_name = self.filepath + str(self.acc) + ".ckpt"
                # ms.save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
                # print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)
            self.start_time = time.time()
    
    def end(self, run_context):
        if(self.train_network is not None):
            file_name = self.filepath + str(self.acc) + ".ckpt"
            ms.save_checkpoint(save_obj=self.train_network, ckpt_file_name=file_name)
            print("Maximum score checkpoint saved")
        else:
            print("no improvement")

class DatasetGenerator:
    def __init__(self,Nf,repeatTimes):
        # print("***************DatasetGenerator __init__*****************")
        # np.random.seed(58)
        self.__index = 0
        self.data_load_address = './data'
        self.Nf=Nf
        # self.Ne=Ne
        # self.steps_per_epoch=steps_per_epoch
        self.repeatTimes=repeatTimes
        self.batch_x=None
        self.batch_y=None

    def __next__(self):
        # if self.__index >= len(self.__data):
        #     raise StopIteration
        # else:
        #     item = (self.__data[self.__index], self.__label[self.__index])
        #     self.__index += 1
        #     return item
        if self.__index >= self.Nf*9000*self.repeatTimes:
            print("***************DatasetGenerator data reading finished*****************")
            raise StopIteration
        if(self.__index%(9000*self.repeatTimes)==0):
            print(self.__index,"reading data from file")
            start_time = time.time()
            dataSetName = os.listdir(self.data_load_address+'/trainSet/')
            while(len(dataSetName)<=1):
                print("data not generated yet")
                time.sleep(5)
                dataSetName = os.listdir(self.data_load_address+'/trainSet/')
            fileIdx=1
            uuidStr=dataSetName[fileIdx].split(".")[0]
            while(dataSetName.count(uuidStr)==0):
                fileIdx=(fileIdx+1)%len(dataSetName)
                if(fileIdx==0):
                    print(uuidStr,"data not ready")
                    time.sleep(0.5)
                    fileIdx=1
                dataSetName = os.listdir(self.data_load_address+'/trainSet/')
                uuidStr=dataSetName[fileIdx].split(".")[0]
            print(uuidStr+'.npy')
            all=np.load(self.data_load_address+'/trainSet/'+uuidStr+'.npy').astype(np.float32)
            os.remove(self.data_load_address+'/trainSet/'+uuidStr)
            os.remove(self.data_load_address+'/trainSet/'+uuidStr+'.npy')
        
            batch_x=all[:2048].T
            batch_y=all[2048:].T
            end_time = time.time()
            print('\nTook %f seconds to read 9000 training data\n' % (end_time - start_time))
            # batch_divider = int(self.steps_per_epoch/self.Ne)#9000/batch_size
            # batch_ys = np.split(batch_y, batch_divider, 0)
            # batch_xs = np.split(batch_x, batch_divider, 0)
            print("repeat dataset")
            batch_ys = []
            batch_xs = []
            for _ in range(self.repeatTimes):
                batch_ys.append(batch_y)
                batch_xs.append(batch_x)
            self.batch_x=np.concatenate(batch_xs,0)
            print(self.batch_x.shape,self.batch_x.dtype)
            self.batch_y=np.concatenate(batch_ys,0)
            print(self.batch_y.shape,self.batch_x.dtype)
        item = (self.batch_y[self.__index%(9000*self.repeatTimes)],self.batch_x[self.__index%(9000*self.repeatTimes)])
        self.__index += 1
        return item

    def __iter__(self):
        # print("***************DatasetGenerator __iter__*****************")
        # self.__index = 0
        return self

    def __len__(self):
        # print("***************DatasetGenerator __len__*****************")
        return self.Nf*9000*self.repeatTimes