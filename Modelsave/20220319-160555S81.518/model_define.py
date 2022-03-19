import tensorflow as tf
import keras
from keras import layers,backend as K
import numpy as np
from tensorflow import math
from keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax
####################定义模型####################
projection_dim = 256
LAYER_NORM_EPS = 1e-6
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 1

def mlp(x, hidden_units, dropout_rate, trainable=True):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu, trainable=trainable)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, trainable):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim,trainable=trainable)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim,trainable=trainable
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            # 'projection': self.projection,
            # 'position_embedding': self.position_embedding,
        })
        return config

def rxModel(input_bits,trainable=True):
    temp = layers.Reshape((256,16*2*2))(input_bits)
    patches = layers.BatchNormalization(trainable=trainable)(temp)
    
    encoded_patches = PatchEncoder(256, projection_dim, trainable=trainable)(patches)
    # encoded_patches = mlp(encoded_patches, hidden_units=[1024,projection_dim], dropout_rate=0,trainable=trainable)

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # if(i==0):
        #     trainable=False
        # else:
        #     trainable=True
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS, trainable=trainable)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0, trainable=trainable
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS, trainable=trainable)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0, trainable=trainable)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    # representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS,trainable=trainable,name="enc_ln_3")(encoded_patches)
    # representation = layers.Flatten()(representation)
    # representation = layers.Dropout(0.1)(representation)
    # Add MLP.
    features = mlp(encoded_patches, hidden_units=[1024,8], dropout_rate=0,trainable=trainable)
    features = layers.Reshape((256,4,2))(features)
    features = layers.Permute((2,1,3))(features)
    temp = layers.Flatten()(features)
    # Classify outputs.
    out_put = layers.Dense(4*256*2, activation='sigmoid',trainable=trainable,name="dense_output")(temp)# 4*256*2 bit
    return out_put