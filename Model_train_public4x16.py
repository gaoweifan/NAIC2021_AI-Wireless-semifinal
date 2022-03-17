import tensorflow as tf
# from tensorflow import keras
# import keras
from keras import optimizers,callbacks,Input,Model
import tensorflow_addons as tfa
from utils import *
from model_define import rxModel
from keras.callbacks import EarlyStopping
from datetime import datetime
import time
import shutil

if __name__=="__main__":
    np.random.seed(int(time.time()))

    ####################定义模型####################
    input_bits = Input(shape=(256*16*2*2,))#256载波*16天线*2（导频/数据）*2（IQ）
    out_put = rxModel(input_bits)
    model=Model(input_bits, out_put)
    model.load_weights('Modelsave/tmp20220317-131930/model_4x16.h5',by_name=True,skip_mismatch=True)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=False)  # 初始学习率为0.001
    # opt = tfa.optimizers.LazyAdam(0.001)
    model.compile(optimizer=opt, loss='mse',metrics=[score_train])
    model.summary()
    
    ###########################以下仅为信道数据载入和链路使用范例############
    import scipy.io as scio

    main_address = '.'
    mat = scio.loadmat('./data/Htrain.mat')
    x_train = mat['H_train']  # shape=?*antennas*delay*IQ
    print(np.shape(x_train))
    H=x_train[:,:,:,0]+1j*x_train[:,:,:,1]

    #############产生Y与X用于验证模型性能，非评测用
    # Y_val, X_val = generatorXY(2000, H)
    Y_val=np.load(main_address+'/data/y_test.npy')[:9000]
    X_val=np.load(main_address+'/data/x_test.npy')[:9000]
    print(np.shape(Y_val))
    print(np.shape(X_val))

    ###############训练###################
    # TensorBoard回调函数
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir_fit = "logs/" + current_time + "/fit"
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir_fit,histogram_freq=1)
    
    # 每轮训练完均测试分数并保存最优权重
    class bestScoreCallback(callbacks.Callback):
        def __init__(self, Y_val, X_val, **kwargs):
            self.x_test = Y_val
            self.y_test = X_val
            self.best_score = 0
            super(bestScoreCallback, self).__init__()

        def on_train_begin(self, logs=None):
            # y_pred = self.model.predict(self.x_test)
            # self.best_score=score_train(self.y_test, y_pred)
            # print("initial best score:",self.best_score)
            return

        def on_epoch_end(self, epoch, logs=None):
            # self.y_test = self.model.predict(self.x_test)
            # NMSE_test=NMSE(self.x_test, self.y_test)
            # tmp_score = Score(NMSE_test)
            tmp_score = logs['val_score_train']
            if(self.best_score<tmp_score):
                print("update best score from",self.best_score,"to",tmp_score)
                self.best_score=tmp_score
                print("saving Model")
                modelpath = f'{main_address}/Modelsave/tmp{current_time}/'
                # encoder.save(modelpath+"encoder.h5")
                # decoder.save(modelpath+"decoder.h5")
                try:
                    os.mkdir(modelpath)
                except:
                    pass
                model.save(modelpath+"model_4x16.h5")
            else:
                print("best score still remain:",self.best_score,",larger than current:",tmp_score)
            return
    bsCallback=bestScoreCallback(Y_val, X_val)

    # 早停回调函数
    esCBk=EarlyStopping(monitor='val_score_train', patience=300, verbose=1, mode='max', baseline=None, restore_best_weights=False)
    
    # each_batch=2000
    # batch_divider=2  # 每批次样本数=each_batch/batch_divider
    # worker=10  # 每轮样本数=each_batch*worker
    # epoch=100  # 总样本数=each_batch*worker*epoch
    # steps_per_epoch=worker*batch_divider
    steps_per_epoch=15
    batch_size=600
    epoch=1500
    repeatTimes=1
    Ne=batch_size*steps_per_epoch/9000  # 每轮需要的文件数
    Nf=int(epoch*Ne/repeatTimes)  # 需要的总文件数
    if (epoch * Ne/repeatTimes != Nf):
        print("Nf必须为整数:", epoch * Ne/repeatTimes)
        exit()
    else:
        print("Nf:", Nf)
        print("Nf per epoch:", Ne)
    if (9000 / batch_size != int(9000 / batch_size)):
        print("batch_divider=9000/batch_size必须为整数:", 9000 / batch_size)
        exit()
    else:
        print("batch_divider:", int(9000 / batch_size))

    model.fit(
        # multiGenerator(H,epoch,each_batch,batch_divider,worker),
        offLineGenerator(H,Nf,Ne,steps_per_epoch,repeatTimes),
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        validation_data=(Y_val, X_val),
        callbacks=[tensorboard_callback,bsCallback,esCBk])
    # model.fit(x=Y, y=X, batch_size=3000, epochs=20, validation_split=0.1,callbacks=tensorboard_callback)

    ##########X与Y的存储方式
    # np.savetxt('Yval.csv', Y, delimiter=',')
    # X.tofile('Xval.bin')

    ############load model and predict Xval_pre from Yval
    # Y_1 = np.loadtxt('Yval.csv', dtype=np.float, delimiter=',')
    # Y_1 = Y
    # Y_1, X_1 = generatorXY(1000, H)
    Y_1, X_1 = Y_val, X_val
    try:
        model.load_weights(f'{main_address}/Modelsave/tmp{current_time}/model_4x16.h5')
    except:
        pass
    Xval_pre = model.predict(Y_1)
    Xval_pre = np.array(np.floor(Xval_pre + 0.5), dtype=bool)
    # Xval_pre.tofile('Xval_pre.bin')

    ##############性能指标
    score_str=str(format(100*np.mean(X_1==Xval_pre), '.3f'))
    print('score=',score_str)

    # 保存模型权重及代码
    modelpath = f'{main_address}/Modelsave/{current_time}S{score_str}/'
    try:
        os.rename(f'{main_address}/Modelsave/tmp{current_time}/', modelpath)
        print("modelpath:",modelpath)
    except:
        print("no improvement")
        model.save(modelpath+'model_4x16.h5')
        print("model saved")
    # save code
    shutil.copyfile(f'{main_address}/model_define.py', modelpath+'model_define.py')