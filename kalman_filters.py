#imports
import numpy as np
import matplotlib.pyplot as plt


class kalman_filter:
    # construct a kalman filter class for furture use
    def __init__(self, Nt, A, Q, H, r, x0_true, P0):
        self.Nt = Nt          # Number of observed data points
        self.A = A              # Coefficient matrix in the evoluation model
        self.Q = Q             # Covariance matrix for the noise in the evoluation model
        self.H = H             # Coefficient matrix in the measurement model
        self.r = r             # Variance of noise in the measurement model
        self.x0_true = x0_true  # Initial values of state variables
        self.mu0 = np.zeros(x0_true.shape[0])  # Predicted mean value for x0
        self.P0 = P0

    def filtering(self, truth_data):
        Nt = self.Nt  # total observation time steps
        H = self.H
        r = self.r
        R = [[r, 0], [0, r]]
        R = np.array(R)
        A = self.A
        Q = self.Q
        x0_true = self.x0_true
        mu0 = self.mu0      # mean for x0
        P0 = self.P0
        measure_ydata = np.zeros((H.shape[0], Nt + 1))
        y0_measure = H @ x0_true + np.random.multivariate_normal(np.zeros(H.shape[0]), R)
        measure_ydata[:, 0] = y0_measure
        for i in range(Nt):
            measure_ydata[:, i + 1] = H @ truth_data[:, i + 1] + np.random.multivariate_normal(np.zeros(H.shape[0]), R)
        # perform kalman filtering
        m_k0 = mu0 + P0 @ (H.T) @ np.linalg.inv(H @ P0 @ (H.T) + R) @ (measure_ydata[:, 0] - H @ mu0)  # initialize m_k-1
        P_k0 = P0 - P0 @ (H.T) @ np.linalg.inv(H @ P0 @ (H.T) + R) @ H @ P0  # initialize P_k-1
        mu_Nt = [m_k0]  # store posterior mean history of state variables
        P_Nt = [P_k0]  # store posterior variance history
        for i in range(Nt):
            # prediction step
            m_km = A @ m_k0
            P_km = Q + A @ P_k0 @ (A.T)

            # update step
            kalman_gain_mat = P_km @ (H.T) @ np.linalg.inv(H @ P_km @ (H.T) + R)
            m_k1 = m_km + kalman_gain_mat @ (measure_ydata[:, i + 1] - H @ m_km)
            P_k1 = P_km - kalman_gain_mat @ H @ P_km
            mu_Nt.append(m_k1)
            P_Nt.append(P_k1)
            m_k0 = m_k1
            P_k0 = P_k1
        kalman_update = np.array(mu_Nt).T
        return (measure_ydata, kalman_update)

    def plotting(self, measure_ydata, kalman_update, truth_data):
        # Plot position predictions
        plt.figure()
        plt.plot(truth_data[0, :], truth_data[1, :], linewidth=2, label='Truth')
        plt.plot(kalman_update[0, :], kalman_update[1, :], linewidth=2, label='Filtered')
        plt.plot(measure_ydata[0, :], measure_ydata[1, :], 'bo', label='Data')
        plt.legend()
        plt.title('Position')
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)

        # Plot velocity predictions
        plt.figure()
        plt.plot(truth_data[2, :], truth_data[3, :], linewidth=2, label='Truth')
        plt.plot(kalman_update[2, :], kalman_update[3, :], linewidth=2, label='Filtered')
        plt.legend()
        plt.title('velocity')
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)

        # Plot acceleration predictions
        plt.figure()
        plt.plot(truth_data[4, :], truth_data[5, :], linewidth=2, label='Truth')
        plt.plot(kalman_update[4, :], kalman_update[5, :], linewidth=2, label='Filtered')
        plt.legend()
        plt.title('acceleration')
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)


class extended_kalman_filter:
    # construct a extended kalman filter class for furture use
    def __init__(self, Nt, A, Q, r, x0_true, P0):
        self.Nt = Nt          # Number of observed data points
        self.A = A              # Coefficient matrix in the evoluation model
        self.Q = Q             # Covariance matrix for the noise in the evoluation model
        self.r = r             # Variance of noise in the measurement model
        self.x0_true = x0_true  # Initial values of state variables
        self.mu0 = x0_true      # Predicted mean value for x0
        self.P0 = P0            # Predicted covariance matrix for x0

    def filtering(self, truth_data):
        Nt = self.Nt  # total observation time steps
        r = self.r
        A = self.A
        Q = self.Q
        x0_true = self.x0_true
        mu0 = self.mu0      # mean for x0
        P0 = self.P0
        measure_ydata = np.zeros(Nt + 1)
        y0_measure = x0_true[2] * np.sin(x0_true[0]) + np.random.normal(loc=0, scale=r)
        measure_ydata[0] = y0_measure
        for i in range(Nt):
            measure_ydata[i + 1] = truth_data[2, i + 1] * np.sin(truth_data[0, i + 1]) + np.random.normal(loc=0,
                                                                                                          scale=r)
        # perform extended kalman filtering
        H = np.array([mu0[2] * np.cos(mu0[0]), 0, np.sin(mu0[0])])
        m_k0 = mu0 + P0 @ (H.T) / (H @ P0 @ (H.T) + r) * (measure_ydata[0] - H @ mu0)  # initialize m_k-1
        P_k0 = P0 - (P0 @ (H.T) / (H @ P0 @ (H.T) + r) @ H) * P0  # initialize P_k-1
        mu_Nt = [m_k0]  # store posterior mean history of state variables
        P_Nt = [P_k0]  # store posterior variance history
        kalman_h = [m_k0[2] * np.sin(m_k0[0])]
        for i in range(Nt):
            # prediction step
            m_km = A @ m_k0
            P_km = Q + A @ P_k0 @ (A.T)
            # update step
            H = np.array([m_km[2] * np.cos(m_km[0]), 0, np.sin(m_km[0])])
            kalman_gain_mat = P_km @ (H.T) / (H @ P_km @ (H.T) + r)
            m_k1 = m_km + kalman_gain_mat * (measure_ydata[i + 1] - m_km[2] * np.sin(m_km[0]))
            P_k1 = P_km - (kalman_gain_mat @ H) * P_km
            mu_Nt.append(m_k1)
            P_Nt.append(P_k1)
            m_k0 = m_k1
            P_k0 = P_k1
            kalman_h.append(m_k1[2] * np.sin(m_k1[0]))
        kalman_h = np.array(kalman_h)
        kalman_update = np.array(mu_Nt).T
        return (measure_ydata, kalman_update, kalman_h)

    def plotting(self, t_axis, kalman_h, truth_h, measure_ydata, kalman_update, truth_data):
        plt.figure()
        plt.plot(t_axis, truth_h, linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_h, linewidth=2, label='Filtered')
        plt.plot(t_axis, measure_ydata, 'bo', label='Data')
        plt.legend()
        plt.title('$h_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.figure()
        plt.plot(t_axis, truth_data[0, :], linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_update[0, :], linewidth=2, label='Filtered')
        plt.legend()
        plt.title('$\\theta_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.figure()
        plt.plot(t_axis, truth_data[1, :], linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_update[1, :], linewidth=2, label='Filtered')
        plt.legend()
        plt.title('$\\omega_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.figure()
        plt.plot(t_axis, truth_data[2, :], linewidth=2, label='Truth')
        plt.plot(t_axis, kalman_update[2, :], linewidth=2, label='Filtered')
        plt.legend()
        plt.title('$\\alpha_{k}$', fontsize=18)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)