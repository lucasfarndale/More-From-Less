import tensorflow as tf
import numpy as np

# Off diagonal, normalize_repr, compute_loss are for Barlow Twins, n_branch_loss is for VICReg
# Set barlow_twins_flag=True to use Barlow Twins, False for VICReg
# MultipleLossCallback is redundant

@tf.function
def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n-1, n+1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])

@tf.function
def normalize_repr(z):
    z_norm = (z - tf.reduce_mean(z, axis=0))/(tf.math.reduce_std(z, axis=0)+1e-4)
    return z_norm

@tf.function
def compute_loss(z_a, z_b, lambd):
    # Get batch size and representation dimension.
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    # Normalize the representations along the batch dimension.
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    # Cross-correlation matrix.
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size

    # Loss.
    on_diag = tf.linalg.diag_part(c) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2))

    off_diag = off_diagonal(c)
    off_diag = tf.reduce_sum(tf.pow(off_diag, 2))

    loss = on_diag + (lambd * off_diag)

    return loss, on_diag, off_diag

mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
@tf.function
def n_branch_loss(z_list, lambd=25, mu=25, nu=1, epsilon=1e-5, gamma=1):
    #Invariance Loss.
    inv_loss=0
    for i, z_a in enumerate(z_list):
        for j, z_b in enumerate(z_list):
            if i!=j:
                inv_loss+=mse(z_a, z_b)

    #Variance Loss.
    stds = [tf.math.sqrt(tf.math.reduce_variance(z, axis=0)+epsilon) for z in z_list]
    var_loss = [tf.reduce_mean(tf.keras.activations.relu(gamma-std_z_a)) for std_z_a in stds]
    var_loss = tf.reduce_sum(var_loss)

    #Covariance Loss.
    cov_loss = 0
    for i, z in enumerate(z_list):
        z -= tf.reduce_mean(z, axis=0)
        cov_z = tf.matmul(tf.transpose(z),z)/tf.cast(tf.shape(z)[0], tf.float32)
        cov_z = tf.linalg.set_diag(cov_z, tf.zeros(cov_z.shape[0:-1]))
        cov_z = tf.reduce_sum(cov_z**2)/z.shape[-1]
        cov_loss += cov_z

    #Total Loss.
    loss = lambd*inv_loss+mu*var_loss+nu*cov_loss

    return loss, inv_loss, var_loss, cov_loss

class MultipleLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        print(f"inv_loss: {self.model.inv_tracker.result()}, var_loss: {self.model.var_tracker.result()}, cov_loss: {self.model.cov_tracker.result()}")


class GeneralMultipleVICReg(tf.keras.Model):
    def __init__(self,
                 encoder_list,              # list of encoders
                 projector_list=None,       # list of projectors
                 encoder_indices=None,      # indices of the encoder to use on each branch, e.g. if encoder_list = [enc1, enc2], set encoder_indices=[0,0,1] to have three branches, with enc1 on the first 2 and enc2 on the 3rd
                 projector_indices=None,    # as with encoder indices, additionally replace index with skip to have no projector on that branch
                 train_flags=None,
#                 lambd=5e-3,                # VICReg/Barlow Twins Loss Parameter - If Barlow Twins, lambd is the weight between on and off diagonal entries of the cross correlation matrix
                 barlow_twins_flag=False,   # False for VICReg loss, True for Barlow Twins loss
                 lambd=25,                  # VICReg loss parameters
                 mu=25,                     # loss = lambd*[invariance term] + mu*[variance term] + nu*[covariance term]
                 nu=1,                      # gamma is parameter for variance hinge loss
                 epsilon=1e-5,              # epsilon is a small value to prevent numerical instabilities
                 gamma=1
                ):
        super(GeneralMultipleVICReg, self).__init__()
        self.encoder_list      = encoder_list
        self.encoder_indices   = encoder_indices
        self.lambd             = lambd
        self.mu                = mu
        self.nu                = nu
        self.epsilon           = epsilon
        self.gamma             = gamma
        self.loss_tracker      = tf.keras.metrics.Mean(name="loss")
        self.barlow_twins_flag = barlow_twins_flag
        
        if self.barlow_twins_flag:
            self.on_diag_tracker  = tf.keras.metrics.Mean(name="on_diag_loss")
            self.off_diag_tracker = tf.keras.metrics.Mean(name="off_diag_loss")
        else:
            self.inv_tracker       = tf.keras.metrics.Mean(name="inv_loss")
            self.var_tracker       = tf.keras.metrics.Mean(name="var_loss")
            self.cov_tracker       = tf.keras.metrics.Mean(name="cov_loss")
        
        self.projector_list    = projector_list
        self.projector_indices = projector_indices
        
        if train_flags is not None:
            self.train_flags = train_flags
        else:
            self.train_flags = [[True for _ in l] for l in [self.encoder_indices, self.projector_indices]]
        self.nested_model_list   = [self.encoder_list, self.projector_list]
        
        if len(self.encoder_indices)!=len(self.projector_indices):
            self.projector_indices+=['skip']*(len(self.encoder_indices)-len(self.projector_indices))
            
        if self.encoder_indices is None:
            self.encoder_indices = range(len(encoder_list))
        if self.projector_indices is None:
            self.projector_indices = range(len(projector_list))

    @property
    def metrics(self):
        if self.barlow_twins_flag:
            return [self.loss_tracker, self.on_diag_tracker, self.off_diag_tracker]
        else:
            return [self.loss_tracker, self.inv_tracker, self.var_tracker, self.cov_tracker]

    def train_step(self, data):

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z_list = []
            for i, ds in enumerate(list(data)):
                z = self.encoder_list[self.encoder_indices[i]](ds, training=True)
                z_list.append(z)
            
            y_list = []
            if self.projector_list:
                for i, ds in enumerate(list(z_list)):
                    if self.projector_indices[i]=='skip':
                        y_list.append(ds)
                    else:
                        y = self.projector_list[self.projector_indices[i]](ds, training=True)
                        y_list.append(y)
            else:
                y_list = z_list
            
            if self.barlow_twins_flag:
                loss, on_diag, off_diag = compute_loss(y_list[0], y_list[1], lambd=self.lambd)
                
            else:
                loss, inv_loss, var_loss, cov_loss = n_branch_loss(y_list, lambd=self.lambd, mu=self.mu, nu=self.nu, epsilon=self.epsilon, gamma=self.gamma)

            
        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, [[self.nested_model_list[j][i].trainable_variables for i in list(set(indices)) if isinstance(i, int)] for j, indices in enumerate([self.encoder_indices, self.projector_indices])])
        gradients = [[[tf.where(tf.math.is_nan(ggg),0.00001,ggg) for ggg in gg] for gg in g] for g in gradients]
        for i in list(set(self.encoder_indices)):
            if i != "skip":
                if self.train_flags[0][i]:
                    self.optimizer[0][i].apply_gradients(zip(gradients[0][i], self.encoder_list[i].trainable_variables))
        for i in list(set(self.projector_indices)):
            if i != "skip":
                if self.train_flags[1][i]:
                    self.optimizer[1][i].apply_gradients(zip(gradients[1][i], self.projector_list[i].trainable_variables))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        if self.barlow_twins_flag:
            self.on_diag_tracker.update_state(on_diag)
            self.off_diag_tracker.update_state(off_diag)
        else:
            self.inv_tracker.update_state(inv_loss)
            self.var_tracker.update_state(var_loss)
            self.cov_tracker.update_state(cov_loss)
        return {m.name: m.result() for m in self.metrics}
