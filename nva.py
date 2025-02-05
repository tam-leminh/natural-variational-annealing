import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Generalized matrix inverse
def G_fun(S, epsi=0):
    if epsi == 0:
        try:
            inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(S)
    else:
        di = S.shape[0]
        inv = np.linalg.inv(np.eye(di) * epsi + np.transpose(S) @ S) @ np.transpose(S)
    return inv

# Plot a Gaussian ellipse given mean and covariance
def plot_gaussian_ellipse(mean, cov, n_std=2.0, ax='none', facecolor='none', **kwargs):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=facecolor, edgecolor='red', **kwargs)
    ax.add_patch(ellipse)

# Compute gradient and Hessian of a mixture model
def GradLogMixt(KMix, piMix, meanMix, covMix, precMix, xi):
    d = meanMix.shape[1]
    
    resp = []
    gradvect = np.zeros(d)
    Hsurp = np.zeros((d, d))
    
    for k in range(KMix):
        Sm_xik = precMix[k,:,:] @ np.transpose((meanMix[k, :] - xi))
        resp_k = piMix[k] * stats.multivariate_normal.pdf(xi, mean=meanMix[k, :], cov=covMix[k,:,:], allow_singular=True)
        resp.append(resp_k)
        gradvect += resp_k * Sm_xik
        Hsurp += resp_k * (np.outer(Sm_xik, Sm_xik) - precMix[k,:,:])
    
    resp_sum = np.sum(resp)
    gradvect /= resp_sum
    Hsurp /= resp_sum
    gradmat = Hsurp - np.outer(gradvect, gradvect)
    
    return {"gradvect": gradvect, "gradmat": gradmat}

# Compute prefixes for a point
def compute_gradient_prefix(mean, prec, xi, pre_cov=True):
    Sm = prec @ (xi - mean)
    if pre_cov:
        return {"pre_exp": Sm, "pre_cov": (np.outer(Sm, Sm) - prec)}
    else:
        return {"pre_exp": Sm}
    
# Compute prefixes for many points
def blackbox_prefixes(mean, prec, xik, pre_cov=True, n_parents=0):
    if n_parents == 0:
        n_parents = xik.shape[0]
    if pre_cov:
        prefixes = [compute_gradient_prefix(mean, prec, xik[n, :], pre_cov=True) for n in range(n_parents)]
        all_pre_mean = np.array([prefix['pre_exp'] for prefix in prefixes])
        all_pre_cov = np.array([prefix['pre_cov'] for prefix in prefixes])
        return all_pre_mean, all_pre_cov
    prefixes = [compute_gradient_prefix(mean, prec, xik[n, :], pre_cov=False) for n in range(n_parents)]
    all_pre_mean = np.array([prefix['pre_exp'] for prefix in prefixes])
    return all_pre_mean
    
# Compute black-box gradient for the mean update
def blackbox_gradient_mu(all_pre_mean, f_values, is_weights=None, ret_indiv_grad=False):
    n_parents = all_pre_mean.shape[0]
    if is_weights is None:
        grad_mu_mc = all_pre_mean * f_values[:n_parents]
    else:
        grad_mu_mc = all_pre_mean * f_values[:n_parents] * is_weights[:n_parents]
    if ret_indiv_grad:
        return np.mean(grad_mu_mc, axis=0), grad_mu_mc
    return np.mean(grad_mu_mc, axis=0)
    
# Compute black-box gradient for the covariance update
def blackbox_gradient_cov(all_pre_cov, f_values, is_weights=None):
    n_parents = all_pre_cov.shape[0]
    if is_weights is None:
        grad_s1_mc = all_pre_cov * f_values[:n_parents, np.newaxis]
    else:
        grad_s1_mc = all_pre_cov * f_values[:n_parents, np.newaxis] * is_weights[:n_parents, np.newaxis]
    return np.mean(grad_s1_mc, axis=0)

# Sample from a multivariate Gaussian
def multivariate_normal_rvs(mean, cov, size):
    result = stats.multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    
    if np.isscalar(mean) or len(mean) == 1:
        return result.reshape(size, 1)
    return result

# Density of a multivariate Gaussian mixture
def dMVNmixture2(x, pis, means, covs, log=True):
    den = sum(pis[k] * stats.multivariate_normal.pdf(x, mean=means[k,:], cov=covs[k, :, :], allow_singular=True)
              for k in range(len(pis)))
    if log:
        if den == 0:
            return -743
        return max(np.log(den), -743)
    return den

# Sample from a multivariate Gaussian mixture
def rMVNmixture(nb, pis, means, covs):
    Kvar = pis.shape[0]
    select_comp = np.random.choice(Kvar, nb, p=pis)
    xik = np.array([ stats.multivariate_normal.rvs(mean=means[select_comp[n], :], cov=covs[select_comp[n], :, :], size=1) for n in range(nb) ])
    return xik

# Parse function arguments for NVA
def parse_init_param(N_iter, init_mixture_param):
    piInit, meanInit, covInit, precInit = init_mixture_param[:4]

    Kvar = len(piInit)
    meanvect = np.tile(meanInit[:, :, np.newaxis], (1, 1, N_iter))
    precvect = np.tile(precInit[:, :, :, np.newaxis], (1, 1, 1, N_iter))
    invprec = np.tile(covInit[:, :, :, np.newaxis], (1, 1, 1, N_iter))
    pivect = np.tile(piInit[:, np.newaxis], (1, N_iter))
    vvect = np.tile(np.append(np.log(piInit[:-1]) - np.log(piInit[-1]), 0)[:, np.newaxis], (1, N_iter))

    return Kvar, meanvect, precvect, invprec, pivect, vvect

# Compute the penalized objective
def objective_w_entropy(target, wKL, pis, means, covs, xik, ret_sorted_xik=False):
    mb_size = xik.shape[0]
    lxik = np.array([target(xik[n, :]) for n in range(mb_size)]).reshape(-1, 1)
    qxik = np.array([dMVNmixture2(xik[n, :], pis, means, covs, log=True) for n in range(mb_size)]).reshape(-1, 1)
    fxik = lxik - wKL*qxik
    if ret_sorted_xik:
        parent_indices = np.argsort(fxik, axis=0)[::-1]
        xik = xik[parent_indices.flatten(), :]
        fxik = fxik[parent_indices.flatten(), :]
        qxik = qxik[parent_indices.flatten(), :]
    return fxik, qxik, xik

# Build grids for elevation plot
def initialize_plotter(xmin, xmax, nx, ymin, ymax, ny, target):
    xgrid = np.linspace(xmin, xmax, nx)
    ygrid = np.linspace(ymin, ymax, ny)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    zgrid = np.vectorize(lambda x, y: target(np.array([x, y])))(xgrid, ygrid)
    return xgrid, ygrid, zgrid

# Plot the step-by-step mean trajectory
def component_plotter(meanvect, covs, xik_select, xik_nonselect, gradients, xgrid, ygrid, zgrid, levels=10, comp=0, ellipse_std=1.0):
    Kvar = meanvect.shape[0]
    B0 = xik_select.shape[0]
    it = meanvect.shape[2]
    arrow_width = (np.max(ygrid) - np.min(ygrid))/400
    fig, axes = plt.subplots()
    plt.xlim(np.min(xgrid), np.max(xgrid))
    plt.ylim(np.min(ygrid), np.max(ygrid))
    plt.title(f"i = {it + 1}, k = {comp}")
    plt.contour(xgrid, ygrid, zgrid, levels=levels)

    for a in range(Kvar):
        plt.plot(meanvect[a, 0, :], meanvect[a, 1, :], 'b-', lw=2)
    plt.scatter(meanvect[:, 0, 0], meanvect[:, 1, 0], color='blue', label='Initial Means')
    if ellipse_std > 0:
        plot_gaussian_ellipse(meanvect[comp, :, it-1], covs[comp, :, :], n_std=ellipse_std, ax = axes)
    plt.scatter(xik_select[:, 0], xik_select[:, 1], color='orange')
    plt.scatter(xik_nonselect[:, 0], xik_nonselect[:, 1], color='black')
    max_grad = max([np.linalg.norm(gradients[n,:], ord=2) for n in range(B0)])
    for n in range(B0):
        axes.arrow(meanvect[comp, 0, it-1], meanvect[comp, 1, it-1], gradients[n, 0]/max_grad, gradients[n, 1]/max_grad, color="gold", width=arrow_width)
    plt.show()

# Update the means of the mixture
def mean_update(mean, cov, grad_mu, lr):
    new_mean = mean + lr * cov @ grad_mu
    return new_mean

# Update the precision matrices of the mixture
def precision_update(cov, prec, grad_s1, lr, damping=0):
    new_prec = prec - lr * grad_s1 + (lr**2 / 2) * grad_s1 @ cov @ grad_s1
    new_prec = (new_prec + np.transpose(new_prec))/2
    new_cov = G_fun(new_prec, 0)
    if damping > 0:
        np.fill_diagonal(new_cov, np.diagonal(new_cov) + damping)
        new_prec = G_fun(new_cov, 0)
    return new_cov, new_prec

# Update the weights of the mixture
def weight_update(pis, vs, fs, lr):
    Kvar = pis.shape[0]
    new_vs = np.append(np.clip(vs[:Kvar-1] + lr * (fs[:Kvar-1] - fs[Kvar-1]), -10, 10), 0)
    new_pis = np.exp(new_vs) / np.sum(np.exp(new_vs))
    return new_pis, new_vs

# Compute the importance sampling weights
def compute_is_weights(xik, mean, cov, dens_mixt):
    mb_size = xik.shape[0]
    dens_comp = np.array([stats.multivariate_normal.pdf(xik[n, :], mean=mean, cov=cov) for n in range(mb_size)]).reshape((mb_size, 1))
    return dens_comp/dens_mixt

# Update the means and covariance matrices of the mixture with the black-box method
def blackbox_updates(xik, mean, cov, prec, f_values, lr, damping, update_prec, n_parents=0, is_weights=None, ret_grad_for_plot=False):
    if update_prec:
        all_pre_mean, all_pre_cov = blackbox_prefixes(mean, prec, xik, pre_cov=True, n_parents=n_parents)
        grad_s1 = blackbox_gradient_cov(all_pre_cov, f_values, is_weights)
        new_cov, new_prec = precision_update(cov, prec, grad_s1, lr, damping)
    else:
        all_pre_mean = blackbox_prefixes(mean, prec, xik, pre_cov=False, n_parents=n_parents)
        new_cov, new_prec = cov, prec

    if ret_grad_for_plot:
        grad_mu, grad_mu_mc = blackbox_gradient_mu(all_pre_mean, f_values, is_weights, ret_indiv_grad=True)
    else:
        grad_mu = blackbox_gradient_mu(all_pre_mean, f_values, is_weights, ret_indiv_grad=False)
        
    new_mean = mean_update(mean, new_cov, grad_mu, lr)

    if ret_grad_for_plot:
        return new_mean, new_cov, new_prec, grad_mu_mc
    return new_mean, new_cov, new_prec

# Update the means and covariance matrices of the mixture with the Hessian method
def hessian_updates(xik, comp_number, pis, means, covs, precs, target_grad, target_hess, lr, wKL, damping, update_prec, is_weights=None, ret_grad_for_plot=False):
    Kvar = pis.shape[0]
    mb_size = xik.shape[0]
    mean, cov, prec = means[comp_number, :], covs[comp_number, :, :], precs[comp_number, :, :]
            
    grad_entropy = [ 
        GradLogMixt(Kvar, pis, means, covs, precs, xik[n, :]) 
        for n in range(mb_size)
    ]
    
    if update_prec:
        hess_lxik = np.array([target_hess(xik[n, :]) for n in range(mb_size)])
        if is_weights is None:
            gradI2 = np.mean(hess_lxik, axis=0)
            gradlogq2 = np.mean([grad_entropy[n]['gradmat'] for n in range(mb_size)], axis=0)
        else:
            gradI2 = np.mean(is_weights[:, np.newaxis] * hess_lxik, axis=0)
            gradlogq2 = np.mean(is_weights[:, np.newaxis] * [grad_entropy[n]['gradmat'] for n in range(mb_size)], axis=0)
        ghat = gradI2 - wKL * gradlogq2
        new_prec = prec - lr * ghat + (lr ** 2 / 2) * ghat @ cov @ ghat
        new_cov = G_fun(new_prec, 0)
        if damping > 0:
            np.fill_diagonal(new_cov, np.diagonal(new_cov) + damping)
            new_prec = G_fun(new_cov, 0)
    else: 
        new_cov, new_prec = cov, prec
    
    grad_lxik = np.array([target_grad(xik[n, :]) for n in range(mb_size)])
    if is_weights is None:
        gradI = np.mean(grad_lxik, axis=0)
        gradlogq = np.mean([grad_entropy[n]['gradvect'] for n in range(mb_size)], axis=0)
    else:
        gradI = np.mean(is_weights * grad_lxik, axis=0)
        gradlogq = np.mean(is_weights * [grad_entropy[n]['gradvect'] for n in range(mb_size)], axis=0)
    gradf = gradI - wKL * gradlogq
    new_mean = mean + lr * new_cov @ gradf

    return new_mean, new_cov, new_prec

# Run NVA algorithm with black-box method
def nva_bb(N_iter, mb_size, learning, wKL, target, init_mixture_param, burn=0, damping=0, imp_sampler=None, util=None, n_parents=0, verbose=0, plots = 0, levels = 10):
 
    Kvar, meanvect, precvect, invprec, pivect, vvect = parse_init_param(N_iter, init_mixture_param)
    fitness_shaping = (util is not None)
    if fitness_shaping:
        if n_parents == 0:
            n_parents = mb_size
            normalized_util = util[:n_parents].reshape(n_parents, 1)
        else:
            normalized_util = util[:n_parents].reshape(n_parents, 1) * n_parents / mb_size
        f_values = normalized_util
    
    if plots > 0:
        xgrid, ygrid, zgrid = initialize_plotter(-2, 2, 201, -2, 2, 201, target)
    
    for i in range(1, N_iter):
        plot_this_it = plots > 0 and i % plots == 0
        print_this_it = verbose > 0 and i % verbose == 0

        if print_this_it:
            print(i)

        means_cur, covs_cur, precs_cur, pis_cur, vs_cur = meanvect[:, :, i-1], invprec[:, :, :, i-1], precvect[:, :, :, i-1], pivect[:, i-1], vvect[:, i-1]
        w_cur, lr_cur = wKL[i], learning[i]

        if imp_sampler == "sm":
            xik = rMVNmixture(mb_size, pis_cur, means_cur, covs_cur)
            fxik, qxik, xik = objective_w_entropy(target, w_cur, pis_cur, means_cur, covs_cur, xik, ret_sorted_xik=fitness_shaping)
            dens_mixt = np.exp(qxik)
        elif imp_sampler == "fw":
            xik = rMVNmixture(mb_size, np.ones(Kvar)/Kvar, means_cur, covs_cur)
            fxik, qxik, xik = objective_w_entropy(target, w_cur, pis_cur, means_cur, covs_cur, xik, ret_sorted_xik=fitness_shaping)
            dens_mixt = np.array([dMVNmixture2(xik[n, :], np.ones(Kvar)/Kvar, means_cur, covs_cur, log=False) for n in range(mb_size)]).reshape(-1, 1)
        
        fs = np.zeros(Kvar)
        for k in range(Kvar):
            if imp_sampler is None:
                xik = stats.multivariate_normal.rvs(mean=means_cur[k, :], cov=covs_cur[k, :, :], size=mb_size)
                fxik, qxik, xik = objective_w_entropy(target, w_cur, pis_cur, means_cur, covs_cur, xik, ret_sorted_xik=fitness_shaping)
                is_weights = None
                fs[k] = np.mean(fxik)
            else:
                is_weights = compute_is_weights(xik, means_cur[k, :], covs_cur[k, :, :], dens_mixt)
                fs[k] = np.mean(fxik * is_weights)

            if not fitness_shaping:
                f_values = fxik
            
            meanvect[k, :, i], invprec[k, :, :, i], precvect[k, :, :, i], grad_mu_mc = blackbox_updates(xik, means_cur[k, :], covs_cur[k, :, :], precs_cur[k, :, :], 
                                                                                        f_values, lr_cur, damping, update_prec = (i > burn), is_weights=is_weights, 
                                                                                        n_parents=n_parents, ret_grad_for_plot=True)

            if plot_this_it:
                grads_fp = np.array([invprec[k, :, :, i] @ grad_mu_mc[n,:] for n in range(n_parents)])
                component_plotter(meanvect[:, :, 0:(i-1)], covs_cur, xik[0:4,:], xik[4:16,], grads_fp, xgrid, ygrid, zgrid, levels, comp=k, ellipse_std=1.0)

        pivect[:, i], vvect[:, i] = weight_update(pis_cur, vs_cur, fs, lr_cur)
        
        if print_this_it:
            print(pivect[:, i])
            print(meanvect[:, :, i])
    
    return pivect, meanvect, invprec, precvect

# Run NVA algorithm with Hessian method
def nva_hess(N_iter, mb_size, learning, wKL, target_deriv, init_mixture_param, burn=0, damping=0, imp_sampler=None, util=None, n_parents=0, verbose=0, plots = 0, levels = 10):

    target, target_grad, target_hess = target_deriv[:3]

    Kvar, meanvect, precvect, invprec, pivect, vvect = parse_init_param(N_iter, init_mixture_param)
        
    if plots > 0:
        xgrid, ygrid, zgrid = initialize_plotter(-2, 2, 201, -2, 2, 201, target)
    
    for i in range(1, N_iter):
        plot_this_it = plots > 0 and i % plots == 0
        print_this_it = verbose > 0 and i % verbose == 0

        if print_this_it:
            print(i)

        means_cur, covs_cur, precs_cur, pis_cur, vs_cur = meanvect[:, :, i-1], invprec[:, :, :, i-1], precvect[:, :, :, i-1], pivect[:, i-1], vvect[:, i-1]
        w_cur, lr_cur = wKL[i], learning[i]

        if imp_sampler == "sm":
            xik = rMVNmixture(mb_size, pis_cur, means_cur, covs_cur)
            fxik, qxik, xik = objective_w_entropy(target, w_cur, pis_cur, means_cur, covs_cur, xik, ret_sorted_xik=False)
            dens_mixt = np.exp(qxik)
        elif imp_sampler == "fw":
            xik = rMVNmixture(mb_size, np.ones(Kvar)/Kvar, means_cur, covs_cur)
            fxik, qxik, xik = objective_w_entropy(target, w_cur, pis_cur, means_cur, covs_cur, xik, ret_sorted_xik=False)
            dens_mixt = np.array([dMVNmixture2(xik[n, :], np.ones(Kvar)/Kvar, means_cur, covs_cur, log=False) for n in range(mb_size)]).reshape(-1, 1)
        
        fs = np.zeros(Kvar)
        for k in range(Kvar):
            if imp_sampler is None:
                xik = stats.multivariate_normal.rvs(mean=means_cur[k, :], cov=covs_cur[k, :, :], size=mb_size)
                fxik, qxik, xik = objective_w_entropy(target, w_cur, pis_cur, means_cur, covs_cur, xik, ret_sorted_xik=False)
                is_weights = None
                fs[k] = np.mean(fxik)
            else:
                is_weights = compute_is_weights(xik, means_cur[k, :], covs_cur[k, :, :], dens_mixt)
                fs[k] = np.mean(fxik * is_weights)

            meanvect[k, :, i], invprec[k, :, :, i], precvect[k, :, :, i] = hessian_updates(xik, k, pis_cur, means_cur, covs_cur, 
                                                                                        precs_cur, target_grad, target_hess, lr_cur, 
                                                                                        w_cur, damping, update_prec = (i > burn), 
                                                                                        is_weights=is_weights, ret_grad_for_plot=False)
        
        pivect[:, i], vvect[:, i] = weight_update(pis_cur, vs_cur, fs, lr_cur)
        
        if verbose > 1:
            print(pivect[:, i])
            print(meanvect[:, :, i])
    
    return pivect, meanvect, invprec, precvect