import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
import keras
import numpy as np
import datetime


class notMIWAE:

    def __init__(self, X, Xval,
                 n_latent=50, n_hidden=100, n_samples=1,
                 activation=tf.nn.tanh,
                 out_dist='gauss',
                 out_activation=None,
                 learnable_imputation=False,
                 permutation_invariance=False,
                 embedding_size=20,
                 code_size=20,
                 missing_process='selfmask',
                 testing=False,
                 name='/tmp/notMIWAE'):

        # ---- data
        self.Xorg = X.copy()
        self.Xval_org = Xval.copy()
        self.n, self.d = X.shape

        # ---- missing
        self.S = np.array(~np.isnan(X), dtype=np.float32)
        self.Sval = np.array(~np.isnan(Xval), dtype=np.float32)

        if np.sum(self.S) < self.d * self.n:
            self.X = self.Xorg.copy()
            self.X[np.isnan(self.X)] = 0
            self.Xval = self.Xval_org.copy()
            self.Xval[np.isnan(self.Xval)] = 0
        else:
            self.X = self.Xorg
            self.Xval = self.Xval_org

        # ---- settings
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.activation = activation
        self.out_dist = out_dist
        self.out_activation = out_activation
        self.embedding_size = embedding_size
        self.code_size = code_size
        self.missing_process = missing_process
        self.testing = testing
        self.batch_pointer = 0
        self.eps = np.finfo(float).eps

        print("Creating graph...")
        tf.reset_default_graph()

        # ---- input
        with tf.variable_scope('input'):
            self.x_pl = tf.placeholder(tf.float32, [None, self.d], 'x_pl')
            self.s_pl = tf.placeholder(tf.float32, [None, self.d], 's_pl')
            self.n_pl = tf.placeholder(tf.int32, shape=(), name='n_pl')

        if learnable_imputation and not testing:
            self.imp = tf.get_variable('imp', shape=[1, self.d])
            self.in_pl = self.x_pl + (1 - self.s_pl) * self.imp
        elif permutation_invariance and not testing:
            self.in_pl = self.permutation_invariant_embedding()
        else:
            self.in_pl = self.x_pl

        # ---- parameters from encoder
        with tf.variable_scope('encoder'):
            self.q_mu, self.q_log_sig2 = self.encoder(self.in_pl)

        # ---- variational distribution
        q_z = tfp.distributions.Normal(loc=self.q_mu, scale=tf.sqrt(tf.exp(self.q_log_sig2)))

        # ---- sample the latent value
        self.l_z = q_z.sample(self.n_pl)  # shape [n_samples, batch_size, d]
        self.l_z = tf.transpose(self.l_z, perm=[1, 0, 2])  # shape [batch_size, n_samples, d]

        # ---- parameters from decoder
        if out_dist in ['gauss', 'normal', 'truncated_normal']:

            with tf.variable_scope('data_process'):
                mu, std = self.gauss_decoder(self.l_z)

            # ---- p(x|z)
            if out_dist == 'truncated_normal':
                p_x_given_z = tfp.distributions.TruncatedNormal(loc=mu, scale=std, low=0.0, high=1.0)
            else:
                p_x_given_z = tfp.distributions.Normal(loc=mu, scale=std)

            # ---- evaluate x in p(x|z)
            self.log_p_x_given_z = tf.reduce_sum(
                tf.expand_dims(self.s_pl, axis=1) * p_x_given_z.log_prob(tf.expand_dims(self.x_pl, axis=1)), axis=-1)

            self.l_out_mu = mu
            # ---- sample xm from p(x|z)
            self.l_out_sample = p_x_given_z.sample()

        elif out_dist == 'bern':

            with tf.variable_scope('data_process'):
                logits = self.bernoulli_decoder(self.l_z)

            # ---- p(x|z)
            p_x_given_z = tfp.distributions.Bernoulli(logits=logits)

            self.log_p_x_given_z = tf.reduce_sum(
                tf.expand_dims(self.s_pl, axis=1) * p_x_given_z.log_prob(tf.expand_dims(self.x_pl, axis=1)), axis=-1)

            self.l_out_mu = tf.nn.sigmoid(logits)
            # ---- sample xm from p(x|z)
            self.l_out_sample = tf.cast(p_x_given_z.sample(), tf.float32)

        elif out_dist in ['t', 't-distribution']:

            with tf.variable_scope('decoder'):
                mu, log_sig2, df = self.t_decoder(self.l_z)

            # ---- p(x|z)
            p_x_given_z = tfp.distributions.StudentT(loc=mu,
                                                     scale=tf.nn.softplus(log_sig2) + 0.0001,
                                                     df=3 + tf.nn.softplus(df))

            self.log_p_x_given_z = tf.reduce_sum(
                tf.expand_dims(self.s_pl, axis=1) * p_x_given_z.log_prob(tf.expand_dims(self.x_pl, axis=1)), axis=-1)

            self.l_out_mu = mu
            self.l_out_sample = p_x_given_z.sample()

        else:
            print("use 'gauss', 'normal', 'truncated_normal' or 'bern' as out_dist")

        # ---- the missing process
        with tf.variable_scope('missing'):

            # ---- mix x_o with samples of x_m
            self.l_out_mixed = self.l_out_sample * tf.expand_dims(1 - self.s_pl, axis=1) + tf.expand_dims(
                self.x_pl * self.s_pl, axis=1)

            self.logits_miss = self.bernoulli_decoder_miss(self.l_out_mixed)

        # ---- p(s|x)
        self.p_s_given_x = tfp.distributions.Bernoulli(logits=self.logits_miss)  # (probs=self.s + self.eps)

        # ---- evaluate s in p(s|x)
        self.log_p_s_given_x = tf.reduce_sum(self.p_s_given_x.log_prob(tf.expand_dims(self.s_pl, axis=1)), axis=-1)

        # --- evaluate the z-samples in q(z|x)
        q_z2 = tfp.distributions.Normal(loc=tf.expand_dims(self.q_mu, axis=1),
                                       scale=tf.sqrt(tf.exp(tf.expand_dims(self.q_log_sig2, axis=1))))
        self.log_q_z_given_x = tf.reduce_sum(q_z2.log_prob(self.l_z), axis=-1)

        # ---- evaluate the z-samples in the prior
        prior = tfp.distributions.Normal(loc=0.0, scale=1.0)
        self.log_p_z = tf.reduce_sum(prior.log_prob(self.l_z), axis=-1)

        # ---- notMIWAE:
        self.notMIWAE = self.get_notMIWAE(self.log_p_x_given_z,
                                          self.log_p_s_given_x,
                                          self.log_q_z_given_x,
                                          self.log_p_z)
        # ---- MIWAE for test-set LLH
        self.MIWAE = self.get_MIWAE(self.log_p_x_given_z,
                                    self.log_q_z_given_x,
                                    self.log_p_z)

        # ---- loss
        if self.testing:
            self.loss = - self.MIWAE
        else:
            self.loss = - self.notMIWAE

        # ---- training stuff
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.global_step = tf.Variable(initial_value=0, trainable=False)

        self.optimizer = tf.train.AdamOptimizer()
        if self.testing:
            tvars = tf.trainable_variables(scope='encoder')
        else:
            tvars = tf.trainable_variables()
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step, var_list=tvars)

        self.sess.run(tf.global_variables_initializer())

        if permutation_invariance:
            svars = tf.trainable_variables('data_process')
            svars.append(self.global_step)
            self.saver = tf.train.Saver(svars)
        else:
            self.saver = tf.train.Saver()

        tf.summary.scalar('Evaluation/loss', self.loss)
        tf.summary.scalar('Evaluation/pxz', tf.reduce_mean(self.log_p_x_given_z))
        tf.summary.scalar('Evaluation/psx', tf.reduce_mean(self.log_p_s_given_x))
        tf.summary.scalar('Evaluation/qzx', tf.reduce_mean(self.log_q_z_given_x))
        tf.summary.scalar('Evaluation/pz', tf.reduce_mean(self.log_p_z))

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.FileWriter(name + '/tensorboard/notmiwae_train/{}/'.format(timestamp),
                                                  self.sess.graph)
        self.val_writer = tf.summary.FileWriter(name + '/tensorboard/notmiwae_val/{}/'.format(timestamp),
                                                self.sess.graph)
        self.summaries = tf.summary.merge_all()

    def encoder(self, x):

        x = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_enc1')(x)
        x = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_enc2')(x)

        mu = keras.layers.Dense(units=self.n_latent, activation=None, name='q_mu')(x)

        log_sig2 = keras.layers.Dense(units=self.n_latent, activation=lambda x: tf.clip_by_value(x, -10, 10),
                                   name='q_log_sigma')(x)

        return mu, log_sig2

    def gauss_decoder(self, z):

        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_dec_gauss1')(z)
        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_dec_gauss2')(z)

        mu = keras.layers.Dense(units=self.d, activation=self.out_activation, name='mu')(z)

        std = keras.layers.Dense(units=self.d, activation=tf.nn.softplus, name='std')(z)

        return mu, std

    def bernoulli_decoder(self, z):

        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_dec_bern1')(z)
        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_dec_bern2')(z)

        logits = keras.layers.Dense(units=self.d, activation=None, name='logits')(z)

        # ---- return logits since it goes better with tfp bernoulli
        return logits

    def t_decoder(self, z):

        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, kernel_initializer='orthogonal', name='l_dec1')(z)
        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, kernel_initializer='orthogonal', name='l_dec2')(z)

        mu = keras.layers.Dense(units=self.d, activation=self.out_activation, kernel_initializer='orthogonal', name='mu')(z)

        log_sigma = keras.layers.Dense(units=self.d, activation=lambda x: tf.clip_by_value(x, -10, 10),
                                       kernel_initializer='orthogonal',
                                       name='log_sigma')(z)

        df = keras.layers.Dense(units=self.d, activation=None, kernel_initializer='orthogonal', name='df')(z)

        return mu, log_sigma, df

    def bernoulli_decoder_miss(self, z):

        if self.missing_process == 'selfmasking':

            self.W = tf.get_variable('W', shape=[1, 1, self.d])
            self.b = tf.get_variable('b', shape=[1, 1, self.d])

            logits = - self.W * (z - self.b)

        elif self.missing_process == 'selfmasking_known':

            self.W = tf.get_variable('W', shape=[1, 1, self.d])
            self.W = tf.nn.softplus(self.W)
            self.b = tf.get_variable('b', shape=[1, 1, self.d])

            logits = - self.W * (z - self.b)

        elif self.missing_process == 'linear':

            logits = keras.layers.Dense(units=self.d, activation=None, name='y')(z)

        elif self.missing_process == 'nonlinear':

            z = keras.layers.Dense(units=self.n_hidden, activation=tf.nn.tanh, name='y')(z)
            logits = keras.layers.Dense(units=self.d, activation=None, name='y')(z)

        else:
            print("use 'selfmasking', 'selfmasking_known', 'linear' or 'nonlinear' as 'missing_process'")
            logits = None

        # ---- return logits since it goes better with tfp bernoulli
        return logits

    def get_ELBO(self, q_z, lpxz):

        self.KL = self.KL_loss(q_z.loc, tf.log(tf.square(q_z.scale)))
        # ---- compare manual KL loss to tf.distributions
        p_z = tf.distributions.Normal(loc=0.0, scale=1.0)
        self.KL_check = tf.reduce_sum(tf.distributions.kl_divergence(q_z, p_z), axis=1)

        # ---- sum over dimensions
        self.KL = tf.reduce_sum(self.KL, axis=-1)

        # ---- mean over the sample dimension
        self.log_p_x_given_z_mean = tf.reduce_mean(lpxz, axis=-1)

        return tf.reduce_mean(self.log_p_x_given_z_mean - self.KL)

    def get_notMIWAE(self, lpxz, lpmz, lqzx, lpz):
        """" the not-MIWAE ELBO """

        # ---- importance weights
        l_w = lpxz + lpmz + lpz - lqzx

        # ---- sum over samples
        log_sum_w = tf.reduce_logsumexp(l_w, axis=1)

        # ---- average over samples
        log_avg_weight = log_sum_w - tf.log(tf.cast(self.n_pl, tf.float32))

        # ---- average over minibatch to get the average llh
        return tf.reduce_mean(log_avg_weight, axis=-1)

    def get_MIWAE(self, lpxz, lqzx, lpz):
        """" the MIWAE ELBO """

        # ---- importance weights
        l_w = lpxz + lpz - lqzx

        # ---- sum over samples
        log_sum_w = tf.reduce_logsumexp(l_w, axis=1)

        # ---- average over samples
        log_avg_weight = log_sum_w - tf.log(tf.cast(self.n_pl, tf.float32))

        # ---- average over minibatch to get the average llh
        return tf.reduce_mean(log_avg_weight, axis=-1)

    def permutation_invariant_embedding(self):
        """https://github.com/microsoft/EDDI"""
        self.E = tf.get_variable('E', shape=[self.d, self.embedding_size])

        # ---- mutliply E and s_pl to zero unobserved dimensions in E
        self.Es = tf.expand_dims(self.s_pl, axis=2) * tf.expand_dims(self.E, axis=0)
        print("Es", self.Es.shape)

        # ---- concatenate with x_pl
        self.Esx = tf.concat([self.Es, tf.expand_dims(self.x_pl, axis=2)], axis=2)
        print("Esx", self.Esx.shape)

        # ---- each 21 dimensional embedding for each of the 784 dimensions needs to go through the same network
        self.Esxr = tf.reshape(self.Esx, [-1, self.embedding_size + 1])
        print("Esxr", self.Esxr.shape)

        # ---- nonlinear mapping h(s_d)
        self.h = keras.layers.Dense(units=self.code_size, activation=tf.nn.relu, name='h1')(self.Esxr)
        print("h", self.h.shape)

        # ---- shape back to reality
        self.hr = tf.reshape(self.h, [-1, self.d, self.code_size])
        print("hr", self.hr.shape)

        # ---- again zero the dimensions with no observations
        # ---- (we might get output in these dimensions due to biases in the neural network)
        self.hz = tf.expand_dims(self.s_pl, axis=2) * self.hr
        print("hz", self.hz.shape)

        # ---- permutation invariant aggregation (summation feature dimension)
        self.g = tf.reduce_sum(self.hz, axis=1)
        print("g", self.g.shape)

        return self.g

    def train_batch(self, batch_size):

        x_batch = self.X[self.batch_pointer: self.batch_pointer + batch_size, :]
        s_batch = self.S[self.batch_pointer: self.batch_pointer + batch_size, :]

        _, _loss, _step = \
            self.sess.run([self.train_op, self.loss, self.global_step],
                          {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: self.n_samples})

        self.tick_batch_pointer(batch_size)

        return _loss

    def val_batch(self):

        batch_size = 100
        val_loss = 0.0
        pxz = 0.0
        psx = 0.0
        pz = 0.0
        qzx = 0.0
        n_val_batches = len(self.Xval) // batch_size

        for i in range(n_val_batches):

            x_batch = self.Xval[i * batch_size: (i + 1) * batch_size]
            s_batch = self.Sval[i * batch_size: (i + 1) * batch_size]

            _loss, _pxz, _psx, _qzx, _pz, _step = \
                self.sess.run([self.loss, self.log_p_x_given_z, self.log_p_s_given_x, self.log_q_z_given_x, self.log_p_z, self.global_step],
                              {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: self.n_samples})

            val_loss += _loss
            pxz += np.mean(_pxz)
            psx += np.mean(_psx)
            pz += np.mean(_pz)
            qzx += np.mean(_qzx)

        val_loss /= n_val_batches
        pxz /= n_val_batches
        psx /= n_val_batches
        pz /= n_val_batches
        qzx /= n_val_batches

        summary = tf.Summary()
        summary.value.add(tag="Evaluation/loss", simple_value=val_loss)
        summary.value.add(tag="Evaluation/pxz", simple_value=pxz)
        summary.value.add(tag="Evaluation/psx", simple_value=psx)
        summary.value.add(tag="Evaluation/qzx", simple_value=qzx)
        summary.value.add(tag="Evaluation/pz", simple_value=pz)

        self.val_writer.add_summary(summary, _step)
        self.val_writer.flush()

        x_batch = self.X[self.batch_pointer: self.batch_pointer + batch_size, :]
        s_batch = self.S[self.batch_pointer: self.batch_pointer + batch_size, :]

        _step, _summaries= \
            self.sess.run([self.global_step, self.summaries],
                          {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: self.n_samples})

        self.train_writer.add_summary(_summaries, _step)
        self.train_writer.flush()

        return val_loss

    def get_llh_estimate(self, Xtest, n_samples=100):
        x_batch = Xtest
        s_batch = (~np.isnan(Xtest)).astype(np.float32)

        _llh = self.sess.run(self.MIWAE,
                          {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: n_samples})

        return _llh

    def tick_batch_pointer(self, batch_size):
        self.batch_pointer += batch_size
        if self.batch_pointer >= self.n - batch_size:
            self.batch_pointer = 0

            try:
                p = np.random.permutation(self.n)
                self.X = self.X[p, :]
                self.S = self.S[p, :]
            except MemoryError as error:
                print("Memory error: no shuffling this time")
                print(error)
            except Exception as exception:
                print("Unexpected exception")
                print(exception)

    def save(self, name):
        print("Saving session...")
        self.saver.save(self.sess, name)

    def load(self, name):
        print("Restoring session...")
        self.saver.restore(self.sess, name)
        print("Session restored from global step ", self.sess.run(self.global_step))

    @staticmethod
    def gauss_loss(x, s, mu, log_sig2):
        """ Gauss as p(x | z) """

        eps = np.finfo(float).eps

        p_x_given_z = - 0.5 * np.log(2 * np.pi) - 0.5 * log_sig2 \
                      - 0.5 * tf.square(x - mu) / (tf.exp(log_sig2) + eps)

        return tf.reduce_sum(p_x_given_z * s, axis=-1)  # sum over d-dimension

    @staticmethod
    def bernoulli_loss(x, s, y):
        eps = np.finfo(float).eps
        p_x_given_z = x * tf.log(y + eps) + (1 - x) * tf.log(1 - y + eps)
        return tf.reduce_sum(s * p_x_given_z, axis=-1)  # sum over d-dimension

    @staticmethod
    def bernoulli_loss_miss(x, y):
        eps = np.finfo(float).eps
        p_x_given_z = x * tf.log(y + eps) + (1 - x) * tf.log(1 - y + eps)
        return tf.reduce_sum(p_x_given_z, axis=-1)  # sum over d-dimension

    @staticmethod
    def KL_loss(q_mu, q_log_sig2):
        KL = 1 + q_log_sig2 - tf.square(q_mu) - tf.exp(q_log_sig2)
        return - 0.5 * tf.reduce_sum(KL, axis=1)


