from tensorflow.keras import layers
def rxModel(input_bits):
    temp = layers.Reshape((256,16,2,2))(input_bits)
    temp = layers.Permute((1,2,4,3))(temp)#256载波*16天线*2（IQ）*2（导频/数据作为通道维）
    temp = layers.BatchNormalization()(temp)

    # temp = layers.Dense(4096, activation='relu')(temp)
    # temp = layers.BatchNormalization()(temp)

    temp = layers.Conv3D(2, (7, 2, 2), padding='same', name="conv3d_1_2")(temp)
    temp = layers.BatchNormalization()(temp)
    temp = layers.LeakyReLU(alpha=0.1)(temp)
    # temp = layers.Activation('relu')(temp)

    temp = layers.Conv3D(8, (7, 2, 2), padding='same', name="conv3d_2_16")(temp)
    temp = layers.BatchNormalization()(temp)
    temp = layers.LeakyReLU(alpha=0.1)(temp)
    # temp = layers.Activation('relu')(temp)

    # temp = layers.Conv3D(64, (7, 2, 2), padding='same', name="conv3d_3_64")(temp)
    # temp = layers.BatchNormalization()(temp)
    # temp = layers.LeakyReLU(alpha=0.1)(temp)
    # # temp = layers.Activation('relu')(temp)

    temp = layers.Conv3D(16, (7, 2, 2), padding='same', name="conv3d_4_16")(temp)
    temp = layers.BatchNormalization()(temp)
    temp = layers.LeakyReLU(alpha=0.1)(temp)
    # temp = layers.Activation('relu')(temp)

    temp = layers.Conv3D(2, (7, 2, 2), padding='same', name="conv3d_5_2")(temp)
    temp = layers.BatchNormalization()(temp)
    temp = layers.LeakyReLU(alpha=0.1)(temp)
    # temp = layers.Activation('relu')(temp)

    temp = layers.Flatten()(temp)
    out_put = layers.Dense(4*256*2, activation='sigmoid')(temp)# 4*256*2 bit
    return out_put