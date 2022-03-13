# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""cifar_resnet50
This sample code is applicable to Ascend.
"""
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
from mindspore.nn import Momentum,Metric
from mindspore import Model, context
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor,TimeMonitor, SummaryCollector
from mindspore import load_checkpoint, load_param_into_net
from model_define_mindspore import getRxNet
from utils import scoreMAE,DatasetGenerator,generator,SaveCallback
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

    epoch_size = args.epoch_size
    net = getRxNet()
    ls = MSELoss(reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01, momentum=0.9)

    model = Model(net, loss_fn=ls, optimizer=opt, metrics={'score':scoreMAE()})

    import scipy.io as scio
    data_home = os.path.join(args.data_url,"Htrain.mat")
    assert os.path.exists(data_home), "the dataset path is invalid!"
    mat = scio.loadmat(data_home)
    x_train = mat['H_train']  # shape=?*antennas*delay*IQ
    H=x_train[:,:,:,0]+1j*x_train[:,:,:,1]
    print(H.shape)

    # as for train, users could use model.train
    if args.do_train:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        dataset = create_dataset(args,H)
        eval_dataset = create_dataset(args,H,training=False)
        # batch_num = dataset.get_dataset_size()
        # config_ck = CheckpointConfig(save_checkpoint_steps=batch_num, keep_checkpoint_max=5)
        # ckpoint_cb = ModelCheckpoint(prefix="model", directory=args.train_url, config=config_ck)
        save_cb = SaveCallback(model,eval_dataset,os.path.join(args.train_url,current_time))
        loss_cb = LossMonitor()
        time_cb = TimeMonitor()
        summary_cb = SummaryCollector(summary_dir=os.path.join('./logs',current_time))
        print("start training")
        model.train(epoch_size, dataset, callbacks=[save_cb, loss_cb,time_cb,summary_cb])

    # as for evaluation, users could use model.eval
    if args.do_eval:
        if args.checkpoint_path:
            param_dict = load_checkpoint(args.checkpoint_path)
            load_param_into_net(net, param_dict)
        eval_dataset = create_dataset(args,training=False)
        res = model.eval(eval_dataset)
        print("result: ", res)


def create_dataset(args,H,training=True):
    """
    create data for next use such as training or inferring
    """
    # dataset_generator = DatasetGenerator(args.numSamples,H,training)
    if(training):
        dataset_generator=generator(args.batch_size,H)
        if args.run_distribute:
            rank_id = int(os.getenv('RANK_ID'))
            rank_size = int(os.getenv('RANK_SIZE'))
            dataset = ds.GeneratorDataset(dataset_generator,["data", "label"],num_parallel_workers=args.data_woker, shuffle=False, num_shards=rank_size, shard_id=rank_id)
        else:
            dataset = ds.GeneratorDataset(dataset_generator,["data", "label"],num_parallel_workers=args.data_woker, shuffle=False)
    else:
        import numpy as np
        Y_val=np.load('./data/y_test.npy')
        X_val=np.load('./data/x_test.npy')
        data = (Y_val, X_val)
        if args.run_distribute:
            rank_id = int(os.getenv('RANK_ID'))
            rank_size = int(os.getenv('RANK_SIZE'))
            dataset = ds.NumpySlicesDataset(data, ["data", "label"], num_parallel_workers=args.data_woker, shuffle=False, num_shards=rank_size, shard_id=rank_id)
        else:
            dataset = ds.NumpySlicesDataset(data, ["data", "label"], num_parallel_workers=args.data_woker, shuffle=False)
    # cifar_ds = ds.Cifar10Dataset(data_home)
    # if args.run_distribute:
    #     rank_id = int(os.getenv('RANK_ID'))
    #     rank_size = int(os.getenv('RANK_SIZE'))
    #     cifar_ds = ds.Cifar10Dataset(data_home, num_shards=rank_size, shard_id=rank_id)

    return dataset
