import numpy as np
import time
import scipy.io as scio
import os

data_load_address = '/code/data'

print("reading data")
start_time = time.time()
dataSetName = os.listdir(data_load_address+'/trainSet/')
uuidStr=dataSetName[0].split(".")[0]
while(dataSetName.count(uuidStr)==0):
    time.sleep(0.5)
    print("data not ready")
print(uuidStr+'.npy')
all=np.load(data_load_address+'/trainSet/'+uuidStr+'.npy')
print(all.shape)
x_train=all[:2048].T
y_train=all[2048:].T
print(x_train.shape)
print(y_train.shape)
end_time = time.time()
print('Took %f seconds to read all training data' % (end_time - start_time))

# start_time = time.time()
# # y_train=np.loadtxt(data_load_address+'/y_train.csv',delimiter=",")
# y_train=np.load(data_load_address+'/y_train.npy')
# # mat = scio.loadmat(data_load_address+'/y_train.mat')
# # y_train = mat['y_train']
# print(y_train.shape)
# end_time = time.time()
# print('Took %f seconds to read y_train' % (end_time - start_time))

# start_time = time.time()
# # x_train=np.loadtxt(data_load_address+'/x_train.csv',delimiter=",")
# x_train=np.load(data_load_address+'/x_train.npy')
# # mat = scio.loadmat(data_load_address+'/x_train.mat')
# # x_train = mat['x_train']
# print(x_train.shape)
# end_time = time.time()
# print('Took %f seconds to read x_train' % (end_time - start_time))

# start_time = time.time()
# # x_bit = np.fromfile(data_load_address+'/x_bit.bin',dtype = bool)
# # x_bit = np.reshape(x_bit,(-1,2048))
# x_bit=np.load(data_load_address+'/x_bit.npy')
# print(x_bit.shape)
# end_time = time.time()
# print('Took %f seconds to read x_bit' % (end_time - start_time))