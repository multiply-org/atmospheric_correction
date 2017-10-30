#mval = [0.2, 0.5, 0.15, 0.15, 0.5, 0.4, 0.3]
from scipy import signal
from scipy.stats import gaussian_kde
from matplotlib import colors
import pylab as plt
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16,8))
ax = ax.ravel()
for i in range(7):
    x = signal.convolve2d(pre_mod.atom.toa[i], np.ones((3,3)), mode='same').ravel()
    y = pre_mod.atom.boa[i].ravel()
    xy = np.vstack([x,y])
    kde = gaussian_kde(xy)(xy)
    ax[i].scatter(x, y, c=kde, s=4, edgecolor='',\
                  norm=colors.LogNorm(vmin=kde.min(), vmax=kde.max()*1.2), cmap = plt.cm.jet,rasterized=True)

    ax[i].set_xlim(0, 0.4)
    ax[i].set_ylim(0, 0.4)
