import numpy as np
import h5py
import tensorflow as tf

# tf-2.x
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.losses import Loss



def get_reconstruction_loss(x, t_decoded):
    '''
    Reconstruction loss for the plain VAE
    '''
    v = 0.1
    # return K.mean(K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2 / (2*v) + 0.5*K.log(2*np.pi*v), axis=-1))
    return K.mean(K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2 / (2*v), axis=-1))
    # return K.sum((K.batch_flatten(x) - K.batch_flatten(t_decoded)) ** 2, axis=-1)


def get_l2_reg_loss(qm):
    # 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    # -0.5 * K.sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=-1)
    # kl = -0.5 * (1 - p_logv + q_logv - K.exp(q_logv) / K.exp(p_logv) - K.square(qm - pm) / K.exp(p_logv))
    l2_reg = 0.5*K.square(qm)
    return K.mean(K.sum(l2_reg, axis=-1))


def get_flux_loss(m, state, state_pred):
    '''
    @params:  state, state_pred shape (batch_size, 60, 60, 2)
              p, p_pred shape (batch_size, 60, 60, 1)
              m shape (batch_size, 60, 60, 1)
    
    @return:  loss_flux: scalar
    
    Only consider discrepancies in total flux, not in phases (saturation not used) 
    '''
    
    perm = K.exp(m)
    p = K.expand_dims(state[:, :, :, 1], -1)
    p_pred = K.expand_dims(state_pred[:, :, :, 1], -1)
    
    tran_x = 1./perm[:, 1:, ...] + 1./perm[:, :-1, ...]
    tran_y = 1./perm[:, :, 1:, ...] + 1./perm[:, :, :-1, ...]
    flux_x = (p[:, 1:, ...] - p[:, :-1, ...]) / tran_x
    flux_y = (p[:, :, 1:, :] - p[:, :, :-1, :]) / tran_y
    flux_x_pred = (p_pred[:, 1:, ...] - p_pred[:, :-1, ...]) / tran_x
    flux_y_pred = (p_pred[:, :, 1:, :] - p_pred[:, :, :-1, :]) / tran_y

    loss_x = K.sum(K.abs(K.batch_flatten(flux_x) - K.batch_flatten(flux_x_pred)), axis=-1)
    loss_y = K.sum(K.abs(K.batch_flatten(flux_y) - K.batch_flatten(flux_y_pred)), axis=-1)

    loss_flux = K.mean(loss_x + loss_y)
    return loss_flux


def get_binary_sat_loss(state, state_pred):
    
    sat_threshold = 0.105
    sat = K.expand_dims(state[:, :, :, 0], -1)
    sat_pred = K.expand_dims(state_pred[:, :, :, 0], -1)
    
    
    sat_bool = K.greater_equal(sat, sat_threshold) #will return boolean values
    sat_bin = K.cast(sat_bool, dtype=K.floatx()) #will convert bool to 0 and 1  
    
    sat_pred_bool = K.greater_equal(sat_pred, sat_threshold) #will return boolean values
    sat_pred_bin = K.cast(sat_pred_bool, dtype=K.floatx()) #will convert bool to 0 and 1  
    
    binary_loss = losses.binary_crossentropy(sat_bin, sat_pred_bin)
    return K.mean(binary_loss)

def get_well_bhp_loss(state, state_pred, prod_well_loc):
    '''
    @params: state: shape (batch_size, 60, 60, 2)
             state_pred: shape (batch_size, 60, 60, 2)
             prod_well_loc: shape (5, 2)
             
    p_true: shape (batch_size, 60, 60, 1)
    p_pred: shape (batch_size, 60, 60, 1)
    
    @return: bhp_loss: scalar
    '''
    
    p_true = K.expand_dims(state[:, :, :, 1], -1)
    p_pred = K.expand_dims(state_pred[:, :, :, 1], -1)
    

    bhp_loss = 0
    for i in range(prod_well_loc.shape[0]):
        bhp_loss += K.mean(K.abs(p_true[:, prod_well_loc[i, 1], prod_well_loc[i, 0], :] - p_pred[:, prod_well_loc[i, 1], prod_well_loc[i, 0], :]))

    
    return bhp_loss

class CustomizedLoss(Loss):
    def __init__(self, 
                 lambda_flux_loss, 
                 lambda_bhp_loss, 
                 lambda_trans_loss):
        
        super(CustomizedLoss, self).__init__()
        self.flux_loss_lambda = lambda_flux_loss
        self.bhp_loss_lambda = lambda_bhp_loss
        self.trans_loss_weight = lambda_trans_loss # The variable 'lambda' in E2C paper Eq. (11)
        
        self.total_loss = None
        self.flux_loss = None
        self.reconstruction_loss = None
        self.well_loss =  None
        
    
    def call(self, xt1, y_pred):
        # Parse y_pred
        xt1_pred, zt1_pred, zt1, zt, xt_rec, xt, perm, prod_loc = y_pred
        
        xt = tf.cast(xt, tf.float32)
        xt1 = tf.cast(xt1, tf.float32)
        
        loss_rec_t = get_reconstruction_loss(xt, xt_rec)
        loss_rec_t1 = get_reconstruction_loss(xt1, xt1_pred)

        loss_flux_t = get_flux_loss(perm, xt, xt_rec) * self.flux_loss_lambda
        loss_flux_t1 = get_flux_loss(perm, xt1, xt1_pred) * self.flux_loss_lambda

        loss_prod_bhp_t = get_well_bhp_loss(xt, xt_rec, prod_loc) * self.bhp_loss_lambda
        loss_prod_bhp_t1 = get_well_bhp_loss(xt1, xt1_pred, prod_loc) * self.bhp_loss_lambda

        loss_l2_reg = get_l2_reg_loss(zt)  # log(1.) = 0.

        loss_bound = loss_rec_t + loss_rec_t1 + \
                     loss_l2_reg  + \
                     loss_flux_t + loss_flux_t1 + \
                     loss_prod_bhp_t + loss_prod_bhp_t1 # JPSE 2020 Gaussian case
        
        # Use zt_logvar to approximate zt1_logvar_pred
        loss_trans = get_l2_reg_loss(zt1_pred - zt1)
        
        self.flux_loss = loss_flux_t + loss_flux_t1
        self.reconstruction_loss = loss_rec_t + loss_rec_t1
        self.well_loss = loss_prod_bhp_t + loss_prod_bhp_t1
        self.total_loss = loss_bound + self.trans_loss_weight * loss_trans
        
        return self.total_loss
    
    def getFluxLoss(self):
        return self.flux_loss
    
    def getReconstructionLoss(self):
        return self.reconstruction_loss
    
    def getWellLoss(self):
        return self.well_loss
    
    def getTotalLoss(self):
        return self.total_loss

