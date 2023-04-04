#imports
import numpy as np
import kalman_filters as kf
import matplotlib.pyplot as plt

#################################### Kalman filter ############################
r = [10, 100, 1000]  # possible values of variance for noise in evolution model
Nt = 100             # total observation time steps
dt = 0.5             # time step size
q = 1                # magnifying factor for noise in measurement model
A = [[1, 0, dt, 0, 0.5 * dt ** 2, 0], [0, 1, 0, dt, 0, 0.5 * dt ** 2],
     [0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt],
     [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
A = np.array(A)      # coefficient matrix in evolution model
Q = [[dt ** 5 / 20, 0, dt ** 4 / 8, 0, dt ** 3 / 6, 0], [0, dt ** 5 / 20, 0, dt ** 4 / 8, 0, dt ** 3 / 6],
     [dt ** 4/8, 0, dt ** 3 / 3, 0, 0.5 * dt ** 2, 0], [0, dt ** 4 / 8, 0, dt ** 3 / 3, 0, 0.5 * dt ** 2],
     [dt ** 3 / 6, 0, 0.5 * dt ** 2, 0, dt, 0], [0, dt ** 3 / 6, 0, 0.5 * dt ** 2, 0, dt]]
Q = np.array(Q) * q   # covariance matrix in measurement model
H = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
H = np.array(H)       # coefficient matrix in measurement model
x0_true = np.array([0, 0, 1, 0.6, 0.4, 0.8])    # true initial values of state variable x0
mu0 = np.zeros(6)  # predictive mean for initial state variable x0
P0_diag = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5])
P0 = np.diag(P0_diag)       # predictive covariance matrix for initial state variable x0

# Generate artificial observation data
truth_data = np.zeros((A.shape[0], Nt + 1))
truth_data[:, 0] = x0_true
for i in range(Nt):
    truth_data[:, i + 1] = A @ truth_data[:, i] + np.random.multivariate_normal(mu0, Q)

# Use Kalman filter to make prediction and plot the predicted values
for i in range(len(r)):
    kf_ins = kf.kalman_filter(Nt, A, Q, H, r[i], x0_true, P0)
    (measure_ydata, kalman_update) = kf_ins.filtering(truth_data)
    kf_ins.plotting(measure_ydata, kalman_update, truth_data)
plt.show()