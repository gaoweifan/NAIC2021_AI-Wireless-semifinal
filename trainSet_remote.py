import numpy as np
from utils import generatorXY,multiGen
import time
# from threading import Thread
from multiprocessing import Process, Manager,Pool,Queue
import uuid
import os
import pysftp
from io import BytesIO

mode=0
SNRdb=10
Pilotnum=8

if __name__=="__main__":
    ###########################以下仅为信道数据载入范例############
    import scipy.io as scio
    print("loading H_train")
    data_load_address = './data'
    mat = scio.loadmat(data_load_address+'/Htrain.mat')
    h_train = mat['H_train']  # shape=?*antennas*delay*IQ
    print(np.shape(h_train))
    H=h_train[:,:,:,0]+1j*h_train[:,:,:,1]

    #############产生Y与X
    print("generating data set")
    while True:
        start_time = time.time()
        dataSet=[generatorXY(9000,H,0)]

        # each_batch=900  # 每批次样本数=each_batch
        # worker=10  # 每轮样本数=each_batch*worker(其中worker=steps_per_epoch)
        # epoch=1  # 总样本数=each_batch*worker*epoch
        # queue = Manager().Queue()
        # getter_process = Process(target=multiGen, args=(each_batch,H,worker,epoch,queue))
        # getter_process.start()
        # dataSet=[]
        # for _ in range(worker*epoch):
        #     dataSet.append(queue.get(True))

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
        print("saving data set")
        
        start_ftime = time.time()
        
        fd=BytesIO()
        arr = np.asanyarray(np.concatenate((x_train.T,y_train.T),0))
        np.lib.format.write_array(fd, arr, allow_pickle=True,pickle_kwargs=dict(fix_imports=True))
        fd.seek(0,0)
        
        fo = BytesIO(b'OK')
        
        end_ftime = time.time()
        
        uuidStr=str(uuid.uuid4())
        print("finish writing",uuidStr,end_ftime - start_ftime)
        
        with pysftp.Connection(host="192.168.168.104", username="607", password="607") as sftp:
            print("Connection succesfully stablished ... ")
            print("Transmitting ... ")
            start_ftime = time.time()

            # Define the remote path where the file will be uploaded
            remoteDirPath = 'E:\\gaowf\\NAICsemi\\github\\data\\trainSet\\'
            remoteFilePath = remoteDirPath+uuidStr
            
            while(len(sftp.listdir(remoteDirPath))>480):#存满了
                print("data full at",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                time.sleep(120)

            sftp.putfo(fd, remoteFilePath+'.npy')
            sftp.putfo(fo, remoteFilePath)
            
            end_ftime = time.time()
            print('Took %f seconds to send data set, speed:%fMB/s' % (end_ftime - start_ftime,fd.getbuffer().nbytes/1048576/(end_ftime - start_ftime)))
        fd.close()
        fo.close()
        del(x_train)
        del(y_train)
        del(dataSet)
        end_time = time.time()
        print('Took %f seconds to generate and transmit data set' % (end_time - start_time))
        print("current generated time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print("next estimated time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()+end_time - start_time)))