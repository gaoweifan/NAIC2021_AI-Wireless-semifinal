import tensorflow as tf
import keras
from keras import layers,backend as K
import numpy as np
from tensorflow import math
from keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax
####################定义模型####################
embed_dim_base = 64
num_heads = 4
mlp_num=1024

class Mlp(layers.Layer):
    def __init__(self, filter_num, drop=0., trainable=True, **kwargs):
        
        super(Mlp, self).__init__(**kwargs)
        
        self.filter_num = filter_num
        self.drop = drop
        
        # MLP layers
        self.fc1 = layers.Dense(filter_num[0],trainable=trainable)
        self.fc2 = layers.Dense(filter_num[1],trainable=trainable)
        
        # Dropout layer
        self.drop = layers.Dropout(drop)
        
        # GELU activation
        self.activation = tf.keras.activations.gelu
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_num': self.filter_num,
            'drop': self.drop,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, x):
        
        # MLP --> GELU --> Drop --> MLP --> Drop
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, trainable=True, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, trainable=trainable)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation=tf.keras.activations.gelu, trainable=trainable), layers.Dense(embed_dim, trainable=trainable),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, trainable=trainable)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, trainable=trainable)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def rxModel(input_bits,trainable=True):
    temp = layers.Reshape((256,16*2*2))(input_bits)
    temp = Mlp([mlp_num,embed_dim_base], drop=0.,trainable=trainable,name="input_mlp")(temp)
    encoded_patches = layers.BatchNormalization(trainable=trainable)(temp)

    # Create multiple layers of the Transformer block.
    
    x1_up = TransformerBlock(embed_dim_base, num_heads, embed_dim_base*2, rate=0.1,trainable=trainable,name="trans_up_1")(encoded_patches)

    x2_up = layers.Dense(embed_dim_base*2, activation=tf.keras.activations.gelu,trainable=trainable)(x1_up)
    x2_up = TransformerBlock(embed_dim_base*2, num_heads, embed_dim_base*4, rate=0.1,trainable=trainable,name="trans_up_2")(x2_up)

    x3_up = layers.Dense(embed_dim_base*4, activation=tf.keras.activations.gelu,trainable=trainable)(x2_up)
    x3_up = TransformerBlock(embed_dim_base*4, num_heads, embed_dim_base*8, rate=0.1,trainable=trainable,name="trans_up_3")(x3_up)

    x3_dn = layers.Dense(embed_dim_base*4, activation=tf.keras.activations.gelu,trainable=trainable)(x3_up)
    x3_dn = TransformerBlock(embed_dim_base*4, num_heads, embed_dim_base*8, rate=0.1,trainable=trainable,name="trans_down_3")(x3_dn)

    x2_dn = layers.Concatenate(-1)([x3_up,x3_dn])
    x2_dn = layers.Dense(embed_dim_base*2, activation=tf.keras.activations.gelu,trainable=trainable)(x2_dn)
    x2_dn = TransformerBlock(embed_dim_base*2, num_heads, embed_dim_base*4, rate=0.1,trainable=trainable,name="trans_down_2")(x2_dn)

    x1_dn = layers.Concatenate(-1)([x2_up,x2_dn])
    x1_dn = layers.Dense(embed_dim_base, activation=tf.keras.activations.gelu,trainable=trainable)(x1_dn)
    x1_dn = TransformerBlock(embed_dim_base, num_heads, embed_dim_base*2, rate=0.1,trainable=trainable,name="trans_down_1")(x1_dn)
    
    x_final = layers.Concatenate(-1)([x1_up,x1_dn])
    x_final = layers.Dense(embed_dim_base, activation=tf.keras.activations.gelu,trainable=trainable)(x_final)
    x_final = TransformerBlock(embed_dim_base, num_heads, embed_dim_base*2, rate=0.1,trainable=trainable,name="trans_final")(x_final)

    # Add MLP.
    features = Mlp([mlp_num,8], drop=0.,trainable=trainable,name="output_mlp")(x_final)
    features = layers.Reshape((256,4,2))(features)
    features = layers.Permute((2,1,3))(features)
    temp = layers.Flatten()(features)
    # Classify outputs.
    out_put = layers.Dense(4*256*2, activation='sigmoid',trainable=trainable,name="dense_output")(temp)# 4*256*2 bit
    return out_put