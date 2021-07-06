from layers import *

# tf-2.x
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Reshape, BatchNormalization
from tensorflow.keras.models import Model
# from keras.layers.merge import Add, Multiply, Concatenate
from tensorflow.keras.layers import Add, Multiply, Concatenate

def create_trans(latent_dim, u_dim):
    '''
    Creates a linear transition model in latent space.

    '''
    zt = Input(shape=(latent_dim,))
    dt = Input(shape=(1,))
    ut = Input(shape=(u_dim,))
    
    zt_expand = Concatenate(axis=-1)([zt, dt])

    trans_encoder = create_trans_encoder(latent_dim + 1)
    hz = trans_encoder(zt_expand)

    At = Dense(latent_dim*latent_dim)(hz)
    At = Reshape((latent_dim, latent_dim))(At)

    Bt = Dense(latent_dim*u_dim)(hz)
    Bt = Reshape((latent_dim, u_dim))(Bt)

    batch_dot_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    
    scalar_multi = Lambda(lambda x: x[0] * x[1]) # Larry Jin
    ut_dt = scalar_multi([ut, dt]) # Larry Jin
    
    zt1 = Add()([batch_dot_layer([At, zt]), batch_dot_layer([Bt, ut_dt])])
    
    # At*zt + Bt*ut

    trans = Model([zt, ut, dt], [zt1])

    return trans


def create_trans_encoder(input_dim):
    '''
    Creates FC transition model.

    '''
    
    zt = Input(shape=(input_dim,))

    # Embed z to hz
    hidden_dim = 200
    hz = fc_bn_relu(hidden_dim)(zt)
    hz = fc_bn_relu(hidden_dim)(hz)
    hz = fc_bn_relu(input_dim-1)(hz) # make sure dim of hz is consistent with 

    trans_encoder = Model(zt, hz)

    return trans_encoder


def create_encoder(latent_dim, input_shape, sigma=0.0):
    '''
    Creates a convolutional encoder model.

    '''
    encoder_input = Input(shape=input_shape, name='image')

    x = conv_bn_relu(16, 3, 3, stride=(2, 2))(encoder_input)
    x = conv_bn_relu(32, 3, 3, stride=(1, 1))(x)
    x = conv_bn_relu(64, 3, 3, stride=(2, 2))(x)
    x = conv_bn_relu(128, 3, 3, stride=(1, 1))(x)

    for i in range(3):
        x = res_conv(128, 3, 3)(x)

    x = Flatten()(x)

    xi_mean = Dense(latent_dim, name='t_mean')(x)
#     t_log_var = Dense(latent_dim, name='t_log_var')(x)
    sampler = create_sampler(sigma)
    xi = sampler(xi_mean)

    return Model(encoder_input, xi, name='encoder')

def create_decoder(latent_dim, input_shape):
    '''
    Creates a (trans-)convolutional decoder model.


    '''
    decoder_input = Input(shape=(latent_dim,), name='t')

    x = Dense(int(input_shape[0] * input_shape[1]/16*128), activation='relu')(decoder_input)

    x = Reshape((int(input_shape[0]/4), int(input_shape[1]/4), 128))(x)

    for i in range(3):
        x = res_conv(128, 3, 3)(x)

    x = dconv_bn_nolinear(128, 3, 3, stride=(1, 1))(x)
    x = dconv_bn_nolinear(64, 3, 3, stride=(2, 2))(x)
    x = dconv_bn_nolinear(32, 3, 3, stride=(1, 1))(x)
    x = dconv_bn_nolinear(16, 3, 3, stride=(2, 2))(x)
    y = Conv2D(input_shape[2], (3, 3), padding='same', activation=None)(x)

    return Model(decoder_input, y, name='decoder')


def sample(t_mean, t_sigma):
    '''
    Draws samples from a standard normal and scales the samples with
    standard deviation of the variational distribution and shifts them
    by the mean.

    Args:
        args: sufficient statistics of the variational distribution.

    Returns:
        Samples from the variational distribution.
    '''
    epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
    return t_mean + t_sigma * epsilon


def create_sampler(t_sigma):
    '''
    Creates a sampling layer.
    '''
    return Lambda(lambda x: sample(x, t_sigma), name='sampler')


def create_e2c(latent_dim, u_dim, input_shape, sigma=0):
    '''
    Creates a E2C.

    Args:
        latent_dim: dimensionality of latent space
        return_kl_loss_op: whether to return the operation for
                           computing the KL divergence loss.

    Returns:
        The VAE model. If return_kl_loss_op is True, then the
        operation for computing the KL divergence loss is
        additionally returned.
    '''

    encoder_ = create_encoder(latent_dim, input_shape, sigma=sigma)
    decoder_ = create_decoder(latent_dim, input_shape)
    transition_ = create_trans(latent_dim, u_dim)

    return encoder_, decoder_, transition_

class E2C(Model):
    def __init__(self, latent_dim, u_dim, input_shape, perm_shape, prod_loc_shape, sigma=0.0):
        super(E2C, self).__init__()
        self._build_model(latent_dim, u_dim, input_shape, sigma)
        self.perm_shape = perm_shape
        self.prod_loc_shape = prod_loc_shape
    
    def _build_model(self, latent_dim, u_dim, input_shape, sigma):
        self.encoder, self.decoder, self.transition = create_e2c(latent_dim, u_dim, input_shape, sigma)
    
    def call(self, inputs):
        self.xt, self.ut, self.dt, self.perm, self.prod_loc = inputs
        
        self.zt = self.encoder(self.xt)
        self.xt_rec = self.decoder(self.zt)
        self.zt1_pred = self.transition([self.zt, self.ut, self.dt])
        self.xt1_pred = self.decoder(self.zt1_pred)
        return self.xt1_pred, self.zt1_pred, self.zt, self.xt_rec, self.perm, self.prod_loc
    
    def loadWeightsFromFile(self, encoder_file, decoder_file, transition_file):
        self.encoder.load_weights(encoder_file)
        self.decoder.load_weights(decoder_file)
        self.transition.load_weights(transition_file)
        
    def saveWeightsToFile(self, encoder_file, decoder_file, transition_file):
        self.encoder.save_weights(encoder_file)
        self.decoder.save_weights(decoder_file)
        self.transition.save_weights(transition_file)
