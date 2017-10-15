from keras.models import Model
from keras.layers import Conv2D, concatenate, BatchNormalization, Input, MaxPooling2D, AveragePooling2D, UpSampling2D, Add, Activation
from keras.layers.merge import Concatenate
import keras

def leaky(alpha=0.1):
    def activation():
        return keras.layers.advanced_activations.LeakyReLU(alpha)
    return activation

def relu():
    return Activation('relu')

def elu():
    return Activation('elu')

def prelu():
    return keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

def block(x, name, kernel=3, filter=4, dilation=1, regularizer=None, activation=relu, BN=False, initializer='glorot_uniform'):
    if kernel == 3:
        n = 3
    elif kernel == 7:
        n = 1
    else:
        raise Exception('kernel must be 3 or 7')

    for idx in range(n):
        x = Conv2D(filter, kernel, padding='same', dilation_rate=dilation, kernel_regularizer=regularizer, name=name+'_conv'+str(idx+1), kernel_initializer=initializer)(x)
        x = activation()(x)
        if BN is True:
            x = BatchNormalization(name=name+'_BN'+str(idx+1))(x)
    return x

def create_model(shape, num_blocks=3, kernel=3, filter=4, encoding_dilation=1, decoding_dilation=1, regularizer=None, activation=relu, BN=False, pooling='max', initializer='glorot_uniform', num_classes=21):
    img_in = Input(shape=shape)
    x = img_in

    # encoding
    encoders = []
    for i in range(num_blocks):
        x = block(x, name='down_block'+str(i+1), kernel=kernel, filter=filter*(2**i), dilation=encoding_dilation, regularizer=regularizer, activation=activation, BN=BN, initializer=initializer)
        encoders.append(x)
        if pooling == 'max':
            x = MaxPooling2D(padding='same')(x)
        elif pooling == 'average':
            x = AveragePooling2D(padding='same')(x)
        else:
            raise Exception('pooling must be "max" or "average"')

    # processing
    x = block(x, name='center_block', kernel=kernel, filter=filter*(2**num_blocks), dilation=encoding_dilation, regularizer=regularizer, activation=activation, BN=BN, initializer=initializer)

    # decoding
    for i in range(num_blocks):
        x = UpSampling2D()(x)
        x = block(x, name='up_block'+str(num_blocks-i), kernel=kernel, filter=filter*(2**(num_blocks-i-1)), dilation=decoding_dilation, regularizer=regularizer, activation=activation, BN=BN, initializer=initializer)
        x = Concatenate(axis=-1)([x, encoders.pop()])
        x = Conv2D(filter*(2**(num_blocks-i-1)), 1,
                   name='up_block'+str(num_blocks-i)+'_concate_conv',
                   kernel_regularizer=regularizer,
                   kernel_initializer=initializer,
                   padding='same')(x)
        x = activation()(x)
        if BN is True:
            x = BatchNormalization(name='up_block'+str(num_blocks-i)+'_concate_BN')(x)

    classify = Conv2D(num_classes, 1, name='classify', activation='softmax')(x)
    return Model(inputs=img_in, outputs=classify)

if __name__ == '__main__':
    model = create_model((256, 256, 3), num_blocks=3, kernel=3, filter=4, dilation=1, regularizer=None, activation=relu, BN=True, pooling='max')
    model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(model, show_shapes=True)

