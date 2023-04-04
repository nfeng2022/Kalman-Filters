#imports
import numpy as np
import kalman_filters as kf
import matplotlib.pyplot as plt


#################################### extended Kalman filter ############################
r = 0.1          # variance for noise in evolution model
Nt = 100           # total observation time steps
dt = 0.01           # time step size
t_axis = np.arange(Nt+1)*dt
q1 = 0.2
q2 = 0.8
A = [[1, dt, 0], [0, 1, 0], [0, 0, 1]]
A = np.array(A)                # coefficient matrix in evolution model
Q = [[q1*dt ** 3 / 3, 0.5*q1*dt**2, 0],
     [0.5*q1*dt**2, dt*q1, 0],
     [0, 0, dt *q2]]
Q = np.array(Q)               # covariance matrix in measurement model
x0_true = np.array([0, 10, 1])  # true initial values of state variable x0
P0_diag = np.array([3, 3, 3])
P0 = np.diag(P0_diag)            # predictive covariance matrix for x0

# Generate artificial observation data
mu0 = np.zeros(3)  # mean for q_k-1
truth_data = np.zeros((A.shape[0], Nt + 1))
truth_data[:, 0] = x0_true
truth_h = [x0_true[2]*np.sin(x0_true[0])]
for i in range(Nt):
    truth_data[:, i + 1] = A @ truth_data[:, i] + np.random.multivariate_normal(mu0, Q)
    truth_h.append(truth_data[2, i + 1]*np.sin(truth_data[0, i + 1]))
truth_h = np.array(truth_h)


# Use extended Kalman filter to make prediction and plot the predicted values
ekf_ins = kf.extended_kalman_filter(Nt, A, Q, r, x0_true, P0)
(measure_ydata, kalman_update, kalman_h) = ekf_ins.filtering(truth_data)
ekf_ins.plotting(t_axis, kalman_h, truth_h, measure_ydata, kalman_update, truth_data)
plt.show()