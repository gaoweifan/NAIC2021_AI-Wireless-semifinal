from keras import layers,backend as K
import tensorflow as tf
####################定义模型####################
base_channels=4
embedding_dim=256
num_heads=8
num_layers=4
transformer_units=[1024,embedding_dim]
def InitConv(x):
    return layers.Conv3D(base_channels, (3, 3, 3), padding='same')(x)

def EnBlock(x,out_channels):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(out_channels, (3, 3, 3), padding='same')(x)

    y = layers.BatchNormalization()(x)
    y = layers.Activation('relu')(y)
    y = layers.Conv3D(out_channels, (3, 3, 3), padding='same')(y)

    y = y + x
    return y

def EnDown(x,out_channels,strides):
    return layers.Conv3D(out_channels, (3, 3, 3), strides=strides, padding='same')(x)

def DeUp_Cat(x,prev,out_channels,strides):
    x = layers.Conv3D(out_channels, (1, 1, 1), padding='same')(x)
    y = layers.Conv3DTranspose(out_channels, (1, 1, 1), strides=strides, padding='same')(x)
    y = K.concatenate([prev,y],-1)
    y = layers.Conv3D(out_channels, (1, 1, 1), padding='same')(y)
    return y

def DeBlock(x,out_channels):
    y = layers.Conv3D(out_channels, (3, 3, 3), padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.Conv3D(out_channels, (3, 3, 3), padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = y + x
    return y

def DeBlock1(x,out_channels):
    x = layers.Conv3D(out_channels, (3, 3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(out_channels, (3, 3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x

def mlp(x, hidden_units, dropout_rate):
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformerBlock(encoded_patches):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embedding_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])
    return encoded_patches

def rxModel(input_bits):
    temp = layers.Reshape((256,16,2,2))(input_bits)
    temp = layers.Permute((1,2,4,3))(temp)#256载波*16天线*2（IQ）*2（导频/数据作为通道维）

    # encoder
    temp=InitConv(temp)                       # (256,16,2,16)

    x1=EnBlock(temp,base_channels)
    temp=EnDown(x1,base_channels*2,(2,2,1)) # (128,8,2,32)

    x2=EnBlock(temp,base_channels*2)
    temp=EnBlock(x2,base_channels*2)
    temp=EnDown(temp,base_channels*4,(2,2,1)) # (64,4,2,64)

    x3=EnBlock(temp,base_channels*4)
    temp=EnBlock(x3,base_channels*4)
    temp=EnDown(temp,base_channels*8,(2,2,2)) # (32,2,1,128)

    x4=EnBlock(temp,base_channels*8)
    temp=EnBlock(x4,base_channels*8)
    temp=EnBlock(temp,base_channels*8)
    temp=EnBlock(temp,base_channels*8)        # (32,2,1,128)

    temp = layers.BatchNormalization()(temp)
    temp = layers.Activation('relu')(temp)
    temp = layers.Conv3D(embedding_dim, (3, 3, 3), padding='same')(temp)# (32,2,1,512)
    temp = layers.Reshape((-1,embedding_dim))(temp)                     # (64,512)

    for _ in range(num_layers):
        temp = transformerBlock(temp)

    # decoder
    temp = layers.Reshape((32,2,1,embedding_dim))(temp)                # (32,2,1,512)
    temp = DeBlock1(temp,base_channels*8)
    temp = DeBlock(temp,base_channels*8)                                # (32,2,1,128)

    temp = DeUp_Cat(temp,x3,base_channels*4,(2,2,2))                    # (64,4,2,64)
    temp = DeBlock(temp,base_channels*4)

    temp = DeUp_Cat(temp,x2,base_channels*2,(2,2,1))                    # (128,8,2,32)
    temp = DeBlock(temp,base_channels*2)

    temp = DeUp_Cat(temp,x1,base_channels,(2,2,1))                      # (256,16,2,16)
    temp = DeBlock(temp,base_channels)

    temp = layers.Conv3D(2, (1, 1, 1), padding='same')(temp)            # (256,16,2,2)
    temp = layers.BatchNormalization()(temp)
    temp = layers.Activation('relu')(temp)
    # temp = layers.Conv3D(2, (3, 3, 3), strides=(1,2,1), padding='same')(temp)# (256,8,2,2)
    # temp = layers.BatchNormalization()(temp)
    # temp = layers.Activation('relu')(temp)
    temp = layers.Conv3D(2, (3, 3, 3), strides=(1,4,1), padding='same')(temp)# (256,4,2,2)
    temp = layers.BatchNormalization()(temp)
    temp = layers.Activation('relu')(temp)
    
    temp = layers.Softmax()(temp)
    temp = temp[:,:,:,:,1]

    temp = layers.Permute((2,1,3))(temp)
    out_put = layers.Flatten()(temp)
    
    return out_put