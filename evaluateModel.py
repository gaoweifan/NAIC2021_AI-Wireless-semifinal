import numpy as np
import scipy.io as scio
from tensorflow.keras import models
from utils import get_custom_objects,MIMO4x16,DequantizationOp
import time

mode=0
SNRdb=10
Pilotnum=8
B=2

#载入信道
print("loading H_train")
data_load_address = './data'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']  # shape=?*antennas*delay*IQ
print(np.shape(x_train))
H=x_train[:,:,:,0]+1j*x_train[:,:,:,1]

#使用链路和信道数据产生训练数据
def generator(batch,H):
    start_time = time.time()

    input_labels = []
    input_samples = []
    for i in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits2 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits3 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1, bits2, bits3]
        # h_idx = np.random.randint(0, len(H))
        h_idx = i%9000
        HH = H[h_idx]
        YY = MIMO4x16(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1, bits2, bits3), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = np.asarray(input_samples)
    bit_x = np.asarray(input_labels)
    batch_x = DequantizationOp(bit_x, B).numpy()#加入解量化
    #print(np.shape(batch_y))# (1000, 16384)
    #print(np.shape(batch_x))# (1000, 2048)

    end_time = time.time()
    print(' Took %f seconds to generate %d samples' % (end_time - start_time,batch))

    return (batch_y, batch_x, bit_x)
print("generating test set")
Y, X, x_bit = generator(1000, H)

#载入模型
model = models.load_model('TrainMIMO/model_4x16.h5', custom_objects=get_custom_objects())
Xval_pre = model.predict(Y)
Xval_pre = np.array(np.floor(Xval_pre + 0.5), dtype=bool)

#性能指标
print('score=',100*np.mean(x_bit==Xval_pre))