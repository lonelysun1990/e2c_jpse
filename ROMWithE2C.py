import tensorflow as tf
from e2c import E2C
from loss import CustomizedLoss

class ROMWithE2C():
    def __init__(self, latent_dim, u_dim, input_shape, perm_shape, prod_loc_shape, learning_rate, 
                 sigma=0.0, lambda_flux_loss=1/1000., lambda_bhp_loss=20, lambda_trans_loss=1.):
        self.model = E2C(latent_dim, u_dim, input_shape, perm_shape, prod_loc_shape, sigma)
        self.loss_object = CustomizedLoss(lambda_flux_loss, lambda_bhp_loss, lambda_trans_loss)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_reconstruction_loss = tf.keras.metrics.Mean(name='train/reconstruction_loss')
        self.train_flux_loss = tf.keras.metrics.Mean(name='train/flux_loss')
        self.train_well_loss = tf.keras.metrics.Mean(name='train/well_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    def predict(self, inputs):
        xt1_pred, zt1_pred, zt, xt_rec, perm, prod_loc = self.model(inputs)
        return  xt1_pred.numpy()
    
    @tf.function
    def evaluate(self, inputs, labels):
        xt1 = labels # y_true
        xt, ut, dt, _, _ = inputs
        
        predictions = self.model(inputs)
        # Parse predictions
        xt1_pred, zt1_pred, zt, xt_rec, perm, prod_loc = predictions
        zt1 = self.model.encoder(xt1)
        y_pred = (xt1_pred, zt1_pred, zt1, zt, xt_rec, xt, perm, prod_loc)
        t_loss = self.loss_object(xt1, y_pred)

        self.test_loss(t_loss)
    
    @tf.function
    def update(self, inputs, labels):
        xt1 = labels # y_true
        xt, ut, dt, _, _ = inputs
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            # Parse predictions
            xt1_pred, zt1_pred, zt, xt_rec, perm, prod_loc = predictions
            zt1 = self.model.encoder(xt1)
            y_pred = (xt1_pred, zt1_pred, zt1, zt, xt_rec, xt, perm, prod_loc)
            loss = self.loss_object(xt1, y_pred)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_flux_loss(self.loss_object.getFluxLoss())
        self.train_reconstruction_loss(self.loss_object.getReconstructionLoss())
        self.train_well_loss(self.loss_object.getWellLoss())