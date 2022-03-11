# NAIC2021_AI-Wireless-semifinal

#### 介绍
NAIC2021比赛，AI+无线通信赛道复赛

### 软件架构
#### model_define.py
为定义模型的代码
#### Model_train_public4x16.py
为训练模型的代码
#### trainSet.py
为生成训练数据集的代码，可多进程并发运行，在data/trainSet文件夹下生成以随机UUID为文件名的npy文件((4\*256\*2)+(256\*16\*2\*2),2\*samples)，将X和Y转置后拼接在一起存到同一个numpy array中。当npy文件写入完成后将会创建一个内容为OK的以相同UUID为文件名的标志文件，用以表示文件写入完成。
#### testSet.py
为生成测试数据集的代码，单线程在data文件夹内生成待发送二进制比特x_train.npy(None,4\*256\*2)和接收数据y_train.npy(None,256\*16\*2\*2)。
#### Modelsave
文件夹内为历次训练的模型权重、及本次训练使用的模型代码定义文件model_define.py。
按照时间及得分划分子文件夹，格式为modelpath = f'./Modelsave/{current_time}S{score_str}/'
#### logs
文件夹内为tensorboard保存的记录文件，可以通过tensorboard --logdir logs启用可视化网页查看
#### data
文件夹内为数据集，包含9000个信道样本Htrain.mat(9000,64,126,2)，测试集和trainSet文件夹内的训练集，训练集命名为随机UUID，包含数据本身与数据写入完成标志，训练过程中会按名称排序挨个读取npy，读取完成后将二者一起删除。
### 环境要求
1.  python==3.9.7
2.  tensorflow-gpu==2.6.0
3.  keras==2.6.0
4.  numpy==1.21.2
5.  cuda>=11.3.1
6.  cudnn>=8.2.1

### 使用说明
1.  先按照要求搭建运行环境，未提及的模块在运行时若报错请自己查询添加
2.  运行testSet.py生成一次测试集
3.  运行trainSet.py连续生成训练集待训练使用
4.  在model_define.py内部定义模型结构
5.  运行Model_train_public4x16.py开始训练，训练完后会自动保存训练权重、模型定义代码到相应文件夹内