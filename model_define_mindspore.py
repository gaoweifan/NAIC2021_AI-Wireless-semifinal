import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore import ops

class rxNet(nn.Cell):
    """
    rxNet网络结构
    """
    def __init__(self):
        super(rxNet, self).__init__()
        # 定义所需要的运算
        self.reshape=ops.Reshape()
        self.transpose=ops.Transpose()
        self.conv1 = nn.Conv3d(2, 2, (7, 2, 2), pad_mode='same')
        self.bn1 = nn.BatchNorm3d(2)
        # self.conv2 = nn.Conv3d(2, 8, (7, 2, 2), pad_mode='same')
        # self.bn2 = nn.BatchNorm3d(8)
        # self.conv3 = nn.Conv3d(8, 16, (7, 2, 2), pad_mode='same')
        # self.bn3 = nn.BatchNorm3d(16)
        # self.conv4 = nn.Conv3d(16, 2, (7, 2, 2), pad_mode='same')
        # self.bn4 = nn.BatchNorm3d(2)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(256*16*2*2, 4*256*2, weight_init=Normal(0.02),activation='sigmoid')


    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.reshape(x,(-1,256,16,2,2))
        x = self.transpose(x,(0,3,1,2,4))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x

def getRxNet():
    return rxNet()