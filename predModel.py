import numpy as np
from keras import Input,Model
from model_define import rxModel

#载入模型
print("loading model")
input_bits = Input(shape=(256*16*2*2,))#256载波*16天线*2（导频/数据）*2（IQ）
out_put = rxModel(input_bits)
model = Model(input_bits, out_put)
model.load_weights('Modelsave/20220320-114700S87.479/model_4x16.h5')

############predict Xval_pre from Yval
print("loading dataset")
# Y = np.loadtxt('Yval.csv', dtype=np.float32, delimiter=',')
Y = np.load('Y.npy')
print("predicting model")
Xval_pre = model.predict(Y)
Xval_pre = np.array(np.floor(Xval_pre + 0.5), dtype=bool)
print("saving result")
Xval_pre.tofile('Xval_pre.bin')