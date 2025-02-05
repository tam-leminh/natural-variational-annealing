import numpy as np
import matplotlib.pyplot as plt
from nva import *

d = 2

# Example: Triangle Mixture
KMM = 3
sig2 = 0.54
piMM = np.array([1/3, 1/3, 1/3])
meanMM = np.array([[0., 1.], [np.cos(np.pi/6), -0.5], [-np.cos(np.pi/6), -0.5]])
covMM = np.repeat(np.diag([sig2] * d)[np.newaxis, :, :], KMM, axis=0) 
precMM = np.repeat(np.linalg.inv(np.diag([sig2] * d))[np.newaxis, :, :], KMM, axis=0) 

# NVA hyperparameters
Kvar = 4
piInit = np.array([1/Kvar for _ in range(Kvar)])
meanInit = np.random.uniform(-2, 2, (Kvar, d))
covfix = 1.
precInit = np.repeat(np.linalg.inv(np.diag([covfix] * d))[np.newaxis, :, :], Kvar, axis=0)
covInit = np.repeat(np.diag([covfix] * d)[np.newaxis, :, :], Kvar, axis=0)

N = 3000
Nt = 4
w0 = 1.
w_decrease_power = 1.
wKL = w0 / np.arange(1, N+1)**w_decrease_power
lrate_0 = 1.e-4
lr_w_power = 0.7
learning = lrate_0 * (w0 / wKL)**lr_w_power
epsilon = 1.e-7
learning = np.minimum(learning, lrate_0/epsilon)
util = np.concatenate([
    4 * Nt * (np.log(Nt + 1) - np.log(np.arange(1, Nt + 1))) / np.sum(np.log(Nt + 1) - np.log(np.arange(1, Nt + 1))),
    np.zeros(3 * Nt)
])

triangle_levels = [-3.5, -3, -2.7, -2.4, -2.2, -2.16, -2.15, np.log(0.1167), np.log(0.1168)]
target = lambda x: dMVNmixture2(x, piMM, meanMM, covMM, log=True)
hyperparameters = { 
    "N_iter": N,
    "mb_size": 4*Nt,
    "burn": 0,
    "damping": epsilon,
    "init_mixture_param": [piInit, meanInit, covInit, precInit]
}
schedules = { 
    "wKL": wKL,
    "learning": learning
}
alg_type_args = { 
    "imp_sampler": None,
    "util": util,
    "n_parents": Nt
}
debug_args = { 
    "verbose": 0,
    "plots": 0,
    "levels": triangle_levels
}

# Running NVA
res = nva_bb(target=target, **hyperparameters, **schedules, **alg_type_args, **debug_args)
pivect, meanvect, invprec, precvect = res

# Plotting
plt.figure()
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title(f"K = {Kvar}")

xgrid = np.arange(-2, 2, 0.02)
ygrid = np.arange(-2, 2, 0.02)
xgrid, ygrid = np.meshgrid(xgrid, ygrid)

target = lambda x: dMVNmixture2(x, piMM, meanMM, covMM, log=True)
zgrid = np.vectorize(lambda x, y: target(np.array([x, y])))(xgrid, ygrid)

plt.contour(xgrid, ygrid, zgrid, levels=[-3.5, -3, -2.7, -2.4, -2.2, -2.16, -2.15, np.log(0.1167), np.log(0.1168)])
for k in range(Kvar):
    plt.plot(meanvect[k, 0, :], meanvect[k, 1, :], 'b-', lw=1)
plt.scatter(meanInit[:, 0], meanInit[:, 1], color='blue', label='Initial Means', zorder=2)
plt.scatter(meanvect[:, 0, N-1], meanvect[:, 1, N-1], color='red', marker='o', label='Final Means', zorder=2)
plt.legend()
plt.show()