import os
import random
import argparse
import mindspore as ms
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.nn import MSELoss
from mindspore.communication import init
from mindspore.nn import Momentum,Metric,Adam
from mindspore import Model, context
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor,TimeMonitor, SummaryCollector
from mindspore import load_checkpoint, load_param_into_net
from model_define_mindspore import getRxNet
from utils import scoreMAE,DatasetGenerator,generator,multiGenerator,SaveCallback
import time

def run(args):
    random.seed(time.time())
    parser = argparse.ArgumentParser(description='Image classification')
    
    
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    
    if args.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)
    
    if not args.do_eval and args.run_distribute:
        context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          all_reduce_fusion_config=[140])
        init()

    # epoch_size = args.epoch_size
    net = getRxNet()
    ls = MSELoss(reduction="mean")
    opt = Adam(net.trainable_params(), learning_rate=args.learning_rate)

    if(args.loadModelName is not None):
        param_dict = load_checkpoint(os.path.join(args.train_url,args.checkpoint_path))
        load_param_into_net(net, param_dict)
        load_param_into_net(opt, param_dict)

    model = Model(net, loss_fn=ls, optimizer=opt, metrics={'score':scoreMAE()})

    # print("loading H_train")
    # import scipy.io as scio
    # data_home = os.path.join(args.data_url,"Htrain.mat")
    # assert os.path.exists(data_home), "the dataset path is invalid!"
    # mat = scio.loadmat(data_home)
    # x_train = mat['H_train']  # shape=?*antennas*delay*IQ
    # H=x_train[:,:,:,0]+1j*x_train[:,:,:,1]
    # print(H.shape)

    # as for train, users could use model.train
    if args.do_train:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        dataset = create_dataset(args)
        eval_dataset = create_dataset(args,training=False)
        # batch_num = dataset.get_dataset_size()
        config_ck = CheckpointConfig(save_checkpoint_steps=args.data_woker, keep_checkpoint_max=5)
        ckpoint_cb = ModelCheckpoint(prefix="model", directory=args.train_url, config=config_ck)
        save_cb = SaveCallback(model,eval_dataset,os.path.join(args.train_url,current_time),10*args.data_woker)
        loss_cb = LossMonitor()
        # time_cb = TimeMonitor()
        summary_cb = SummaryCollector(summary_dir=os.path.join('./logs',current_time))
        print("start training")
        model.train(1, dataset, callbacks=[save_cb, loss_cb,ckpoint_cb,summary_cb],dataset_sink_mode=False)

    # as for evaluation, users could use model.eval
    if args.do_eval:
        if args.checkpoint_path:
            param_dict = load_checkpoint(args.checkpoint_path)
            load_param_into_net(net, param_dict)
        eval_dataset = create_dataset(args,training=False)
        res = model.eval(eval_dataset)
        print("result: ", res)


def create_dataset(args,H=None,training=True):
    """
    create data for next use such as training or inferring
    """
    # dataset_generator = DatasetGenerator(args.numSamples,H,training)
    if(training):
        # dataset_generator=generator(args.batch_size,H)
        # batch_divider=2  # 每批次样本数=each_batch/batch_divider
        # each_batch=args.batch_size*batch_divider
        # worker=args.data_woker  # 每轮样本数=each_batch*worker
        # epoch=args.epoch_size  # 总样本数=each_batch*worker*epoch
        # steps_per_epoch=worker*batch_divider
        # dataset_generator=multiGenerator(H,epoch,each_batch,batch_divider,worker)
        steps_per_epoch=args.data_woker
        # steps_per_epoch=9
        batch_size=args.batch_size
        epoch=args.epoch_size
        Ne=batch_size*steps_per_epoch/9000  # 每轮需要的文件数
        Nf=int(epoch*Ne/args.repeatTimes)  # 需要的总文件数
        if (epoch * Ne / args.repeatTimes != Nf):
            print("Nf必须为整数:", epoch * Ne / args.repeatTimes)
            exit()
        else:
            print("Nf:", Nf)
            print("Nf per epoch:", Ne)
        if (9000 / batch_size != int(9000 / batch_size)):
            print("batch_divider=9000/batch_size必须为整数:", 9000 / batch_size)
            exit()
        else:
            print("batch_divider:", int(9000 / batch_size))
        dataset_generator=DatasetGenerator(Nf,args.repeatTimes)
        if args.run_distribute:
            rank_id = int(os.getenv('RANK_ID'))
            rank_size = int(os.getenv('RANK_SIZE'))
            dataset = ds.GeneratorDataset(dataset_generator,["data", "label"], shuffle=False, num_shards=rank_size, shard_id=rank_id)
        else:
            dataset = ds.GeneratorDataset(dataset_generator,["data", "label"], shuffle=False)
        dataset = dataset.batch(batch_size)
    else:
        import numpy as np
        print("loading test set")
        Y_val=np.load('./data/y_test.npy').astype(np.float32)[:1000]
        print(Y_val.shape,Y_val.dtype)
        X_val=np.load('./data/x_test.npy').astype(np.float32)[:1000]
        print(X_val.shape,X_val.dtype)
        data = (Y_val, X_val)
        # dataset = data
        if args.run_distribute:
            rank_id = int(os.getenv('RANK_ID'))
            rank_size = int(os.getenv('RANK_SIZE'))
            dataset = ds.NumpySlicesDataset(data, ["data", "label"], shuffle=False, num_shards=rank_size, shard_id=rank_id)
        else:
            dataset = ds.NumpySlicesDataset(data, ["data", "label"], shuffle=False)
    # cifar_ds = ds.Cifar10Dataset(data_home)
    # if args.run_distribute:
    #     rank_id = int(os.getenv('RANK_ID'))
    #     rank_size = int(os.getenv('RANK_SIZE'))
    #     cifar_ds = ds.Cifar10Dataset(data_home, num_shards=rank_size, shard_id=rank_id)

    return dataset
