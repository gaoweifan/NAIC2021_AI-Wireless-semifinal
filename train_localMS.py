from mox_parser import get_parser
from main import run

if __name__ == '__main__':
    args = get_parser()
    args.device_target = "Ascend"
    args.device_num = 1
    args.data_url = './data/'
    args.train_url = './Modelsave/'
    args.do_train = True
    args.do_eval = False
    args.epoch_size = 10
    args.batch_size = 1000
    args.data_woker=9
    args.repeatTimes=1
    # args.numSamples = args.epoch_size*args.batch_size*10
    args.checkpoint_path = None
    args.run_distribute=False
    run(args)
