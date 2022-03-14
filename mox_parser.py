# parser.py 
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('-f', type=str, default="./data", help='jupyter.')
    parser.add_argument('--dataset_path', type=str, default="./data", help='Dataset path.')
    parser.add_argument('--data_url',
                        help='path to training/inference dataset folder',
                        default='./data')

    parser.add_argument('--train_url',
                        help='model folder to save/load',
                        default='./model')

    parser.add_argument('--result_url',
                        help='folder to save inference results',
                        default='./result')
    parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute.')
    parser.add_argument('--device_num', type=int, default=1, help='Device num.')
    parser.add_argument('--device_target', type=str, default="Ascend", help='Device choice Ascend or GPU')
    parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
    parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
    parser.add_argument('--epoch_size', type=int, default=2, help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size.')
    # parser.add_argument('--numSamples', type=int, default=10*10*1000, help='number of samples=epoch_size*step_per_epoch*batch_size.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
    parser.add_argument('--data_woker', type=int, default=9, help='number of wokers for generating data set.')
    parser.add_argument('--repeatTimes', type=int, default=1, help='repeat times for generating data set.')
    parser.add_argument('--loadModelName', type=str, default=None, help='model file name to load.')
    args = parser.parse_args()
    return args
