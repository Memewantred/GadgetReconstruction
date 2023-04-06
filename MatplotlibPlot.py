import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import scipy.stats
import matplotlib

class MyParams():
    def __init__(self) -> None:
        self.colors = {"default": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
                       "classic": ['#3b6291', '#943c39', '#779043', '#624c7c', '#388498', '#bf7334', '#3f6899', '#9c403d', '#7d9847', '#675083', '#3b8ba1', '#c97937']}

def DensityPlot(axes, x, y, label, cov_factor=.25, hist=False, bins=20, color_index:int=0, edgecolor=None, histtype="bar"):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    density = scipy.stats.gaussian_kde(y)
    density.covariance_factor = lambda : cov_factor
    density._compute_covariance()
    axes.plot(x, density(x), label=label, color=colors[color_index])
    if hist == True:
        axes.hist(y, bins=bins, density=True, alpha=.5, color=colors[color_index], edgecolor=edgecolor, histtype=histtype)
    axes.set_xlim(x[0], x[-1])
    
def Chi2Plot(axes, x, y, label=None, color_index:int=0, plot=True):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    df, loc, scale = scipy.stats.chi2.fit(y)
    chiDistribution = scipy.stats.chi2(df, loc, scale)
    if plot:
        if label == None:
            label = "$\chi^2(%.2f)$"%(df)
        axes.plot(x, chiDistribution.pdf(x), color=colors[color_index], label=label, linestyle="dashed")
    return df, loc, scale

def NormPlot(axes, x, y, label=None, color_index:int=0, plot=True):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    loc, scale = scipy.stats.norm.fit(y)
    mean = np.mean(y)
    sigma = np.sqrt(np.var(y))
    skew = scipy.stats.skew(y)
    kur = scipy.stats.kurtosis(y)
    normDistribution = scipy.stats.norm(loc, scale)
    if plot:
        if label == None:
            label = "$\mathbb{N}$(%.2f, %.2f)"%(mean, sigma)
        axes.plot(x, normDistribution.pdf(x), color=colors[color_index], label=label, linestyle="dashed")
    return mean, sigma, skew, kur

def JointHeatMapPlot(fig, axes, x, y, bins, normed=True, colorbar=True, lower=0., upper=100., scale=matplotlib.colors.Normalize, colormap=cm.jet):
    H = 0
    xedges = yedges = np.array([])
    try:
        if (x.shape == y.shape):
            Ndim = x.shape[1]
            for i in range(Ndim):
                tmp, xedges, yedges = np.histogram2d(x[:,i], y[:,i], bins=bins, normed=normed)
                H = H + tmp
        else:
            raise ValueError("x and y must have the same shape")
    except:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, normed=normed)
    if colorbar:
        vmax = np.percentile(H, upper)
        vmin = np.percentile(H, lower)
        norm = scale(vmin=vmin, vmax=vmax)
        pcm = axes.imshow(H, cmap=colormap, norm=norm)
        fig.colorbar(pcm, ax=axes, extend='both')
    else:
        axes.imshow(H)
    try:
        N = len(bins)
    except TypeError:
        N = bins
    axes.set_xlim(0, N)
    axes.set_ylim(0, N)
    return H, xedges, yedges

def Show2dSlice(pos, min, max, axis=2, bins=512, fig=None, axes=None, normed=False, colorbar=True, lower=0.5, upper=99.5, scale=matplotlib.colors.Normalize, colormap=cm.viridis):
    pos = pos[(pos[:, axis] > min) & (pos[:, axis] < max)]
    if (fig is not None) and (axes is not None):
        H, xedges, yedges = JointHeatMapPlot(fig, axes, pos[:, (axis+1)%3], pos[:, (axis+2)%3], bins=bins, normed=normed, colorbar=colorbar, lower=lower, upper=upper, scale=scale, colormap=colormap)
    else:
        H, xedges, yedges = np.histogram2d(pos[:, (axis+1)%3], pos[:, (axis+2)%3], bins=bins, normed=normed)
    return H, xedges, yedges