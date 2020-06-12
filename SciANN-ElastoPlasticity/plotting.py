""" 
Description:
    Plotting for von-Mises elasto-plasticity problem: 
      https://arxiv.org/abs/2003.02751
    
Created by Ehsan Haghighat on 6/10/20.
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 1))

def custom_pcolor(AX, X, Y, Z, title="", bar=True, ZLIM=None, **kwargs):
    ZLIM = np.abs(Z).max() if ZLIM is None else ZLIM
    if 'vmin' in kwargs:
        im = AX.pcolor(X, Y, Z, cmap="seismic", **kwargs)
    else:
        im = AX.pcolor(X, Y, Z, cmap="seismic", 
                       norm=Normalize(vmin=-ZLIM, vmax=ZLIM),
                       **kwargs)
    AX.axis("equal")
    AX.axis([X.min(), X.max(), Y.min(), Y.max()])
    AX.get_xaxis().set_ticks([])
    AX.get_yaxis().set_ticks([])
    AX.set_title(title, fontsize=14)
    if bar: 
        clb = plt.colorbar(im, ax=AX)
        clb.formatter.set_powerlimits((0, 0))
    return im

