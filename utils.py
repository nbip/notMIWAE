import numpy as np


def imputationRMSE(model, Xorg, Xz, X, S, L):
    """
    Imputation error of missing data
    """
    N = len(X)

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]

    def imp(model, xz, s, L):
        l_out, log_p_x_given_z, log_p_z, log_q_z_given_x = model.sess.run(
            [model.l_out_mu, model.log_p_x_given_z, model.log_p_z, model.log_q_z_given_x],
            {model.x_pl: xz, model.s_pl: s, model.n_pl: L})

        wl = softmax(log_p_x_given_z + log_p_z - log_q_z_given_x)

        xm = np.sum((l_out.T * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)

        return l_out, wl, xm, xmix

    XM = np.zeros_like(Xorg)

    for i in range(N):

        xz = Xz[i, :][None, :]
        s = S[i, :][None, :]

        l_out, wl, xm, xmix = imp(model, xz, s, L)

        XM[i, :] = xm

        if i % 100 == 0:
            print('{0} / {1}'.format(i, N))

    return np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S)), XM


def not_imputationRMSE(model, Xorg, Xz, X, S, L):
    """
    Imputation error of missing data, using the not-MIWAE
    """
    N = len(X)

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]

    def imp(model, xz, s, L):
        l_out, log_p_x_given_z, log_p_z, log_q_z_given_x, log_p_s_given_x  = model.sess.run(
            [model.l_out_mu, model.log_p_x_given_z, model.log_p_z, model.log_q_z_given_x, model.log_p_s_given_x],
            {model.x_pl: xz, model.s_pl: s, model.n_pl: L})

        wl = softmax(log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x)

        xm = np.sum((l_out.T * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)

        return l_out, wl, xm, xmix

    XM = np.zeros_like(Xorg)

    for i in range(N):

        xz = Xz[i, :][None, :]
        s = S[i, :][None, :]

        l_out, wl, xm, xmix = imp(model, xz, s, L)

        XM[i, :] = xm

        if i % 100 == 0:
            print('{0} / {1}'.format(i, N))

    return np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S)), XM
