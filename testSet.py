# from lightgbm import Dataset
# import tensorflow as tf
# from tensorflow.keras import optimizers,callbacks,Input,Model,layers
import numpy as np
from utils import *
import time
# from threading import Thread
from multiprocessing import Process, Manager,Pool,Queue
import uuid

mode=0
SNRdb=10
Pilotnum=8

if __name__=="__main__":
    ###########################以下仅为信道数据载入范例############
    import scipy.io as scio
    print("loading H_train")
    data_load_address = './data'
    mat = scio.loadmat('./data/Htrain.mat')
    h_train = mat['H_train']  # shape=?*antennas*delay*IQ
    print(np.shape(h_train))
    H=h_train[:,:,:,0]+1j*h_train[:,:,:,1]

    #############产生Y与X
    print("generating data set")
    
    # dataSet=[generatorXY(10000,H,0)]

    each_batch=1000  # 每批次样本数=each_batch
    worker=10  # 每轮样本数=each_batch*worker(其中worker=steps_per_epoch)
    epoch=1  # 总样本数=each_batch*worker*epoch
    queue = Manager().Queue()
    getter_process = Process(target=multiGen, args=(each_batch,H,worker,epoch,queue))
    getter_process.start()
    dataSet=[]
    for _ in range(worker*epoch):
        dataSet.append(queue.get(True))

    y_train=[]
    x_train=[]
    for data in dataSet:
        y_train.append(data[0])
        x_train.append(data[1])
    y_train=np.concatenate(np.array(y_train))
    x_train=np.concatenate(np.array(x_train))
    x_train = np.array(np.floor(x_train + 0.5), dtype=bool)
    print(np.shape(y_train))
    print(np.shape(x_train))
    
    ##########X与Y的存储方式
    print("saving data set y_test")
    start_time = time.time()
    # np.savetxt(data_load_address+'/y_train.csv', y_train, delimiter=',')
    np.save(data_load_address+'/y_test.npy', y_train)
    # scio.savemat(data_load_address+'/y_train.mat', {"y_train":y_train})
    end_time = time.time()
    print('Took %f seconds to save y_test' % (end_time - start_time))

    print("saving data set x_test")
    start_time = time.time()
    # np.savetxt(data_load_address+'/x_train.csv', x_train, delimiter=',')
    np.save(data_load_address+'/x_test.npy', x_train)
    # scio.savemat(data_load_address+'/x_train.mat', {"x_train":x_train})
    end_time = time.time()
    print('Took %f seconds to save x_test' % (end_time - start_time))
    print("done")