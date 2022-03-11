from tensorflow.keras import layers
def rxModel(input_bits):
    temp = layers.Reshape((256,16,2,2))(input_bits)
    temp = layers.Permute((1,2,4,3))(temp)#256载波*16天线*2（IQ）*2（导频/数据作为通道维）
    temp = layers.Reshape((256,16*2,2))(input_bits)#256载波*32（天线/IQ）*2（导频/数据作为通道维）
    temp = layers.BatchNormalization()(temp)

    # temp = layers.Dense(4096, activation='relu')(temp)
    # temp = layers.BatchNormalization()(temp)

    temp = layers.Conv2D(2, (7, 3), padding='same', name="conv2d_1_2")(temp)
    temp = layers.BatchNormalization()(temp)
    temp = layers.LeakyReLU(alpha=0.1)(temp)
    # temp = layers.Activation('relu')(temp)

    temp = layers.Conv2D(8, (7, 3), padding='same', name="conv2d_2_2")(temp)
    temp = layers.BatchNormalization()(temp)
    temp = layers.LeakyReLU(alpha=0.1)(temp)
    # temp = layers.Activation('relu')(temp)

    # temp = layers.Conv2D(2, (7, 3), padding='same', name="conv2d_1_2")(temp)
    # temp = layers.BatchNormalization()(temp)
    # temp = layers.LeakyReLU(alpha=0.1)(temp)
    # # temp = layers.Activation('relu')(temp)

    temp = layers.Conv2D(16, (7, 3), padding='same', name="conv2d_3_2")(temp)
    temp = layers.BatchNormalization()(temp)
    temp = layers.LeakyReLU(alpha=0.1)(temp)
    # temp = layers.Activation('relu')(temp)

    temp = layers.Conv2D(2, (7, 3), padding='same', name="conv2d_4_2")(temp)
    temp = layers.BatchNormalization()(temp)
    temp = layers.LeakyReLU(alpha=0.1)(temp)
    # temp = layers.Activation('relu')(temp)

    temp = layers.Flatten()(temp)
    out_put = layers.Dense(4*256*2, activation='sigmoid')(temp)# 4*256*2 bit
    return out_put