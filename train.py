from mox_parser import get_parser
import moxing as mox
from main import run

if __name__ == '__main__':
    args = get_parser()
    # 以下代码为新增部分，一定要紧跟args = get_parser()这行代码
    # Copy obs_file to local 
    obs_data_url = args.data_url
    args.data_url = '/home/work/modelarts/inputs/data/'
    obs_train_url = args.train_url
    args.train_url = '/home/work/modelarts/outputs/model/'
    try:
        mox.file.copy_parallel(obs_data_url, args.data_url)
        print("Successfully Download {} to {}".format(obs_data_url, args.data_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, args.data_url) + str(e))
    import os
    print(os.listdir(args.data_url))
    run(args)
    # ..........balabala.......... 你们的代码
    print(os.listdir(args.train_url)) 
    
    # 在代码结尾部分加入以下代码
    # Upload model to obs
    try:
        mox.file.copy_parallel(args.train_url, obs_train_url)
        print("Successfully Upload {} to {}".format(args.train_url, obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(args.train_url, obs_train_url) + str(e))
