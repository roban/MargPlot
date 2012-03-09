"""Tools for 2-D plots of multi-variate data with marginal distributions.

The core of this module is marginal_plot, which plots a
two-dimensional distribution of points with 1D marginal histograms
along each axis.

"""

import itertools
import math
import copy

import pylab
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def marginal_plot_allpairs(variables, labels=None, axesdict=None, **kwargs):
    """Run marginal_plot on all pairs in the set `variables`.


    Example
    -------

    >>> import pylab
    >>> c1 = [[1.,0.9,-0.9],[0.9,2.0,-0.2],[-0.9,-0.2, 3.0]]
    >>> data1 = np.random.multivariate_normal([1.0, 2.0, 3.0], c1, 300)
    >>> from margplot import marginal_plot_allpairs
    >>> axesdict = marginal_plot_allpairs(data1.T, 
                                          labels=['x', 'y','z'],
                                          color='r')
    >>> c2 = [[1.,-0.9,0.9],[-0.9,2.0,0.2],[0.9,0.2, 3.0]]
    >>> data2 = np.random.multivariate_normal([1.0, 1.0, 1.0], c2, 300)
    >>> axesdict = marginal_plot_allpairs(data2.T, 
                                          axesdict=axesdict,
                                          labels=['x', 'y','z'],
                                          color='b')


    """
    pairs = itertools.combinations(variables, 2)
    length = int(math.factorial(len(variables)) 
                 / (2. * math.factorial(len(variables) - 2)))

    if labels is None:
        labels = ['variables[%i]' % i for i in xrange(len(variables))]
    else:
        assert len(labels) == len(variables)
    pairlabels = itertools.combinations(labels, 2)

    if axesdict is None:
        axesdict = dict()
    for (i, (pair, pairlabel)) in enumerate(zip(pairs, pairlabels)):
        if not pairlabel in axesdict:
            axesdict[pairlabel] = None
        axesdict[pairlabel] = marginal_plot(pair, 
                                            labels=pairlabel, 
                                            axeslist=axesdict[pairlabel],
                                            **kwargs)
    return axesdict

def marginal_plot_pairs(xvar, yvars, axeslists=None, 
                        xlabel=None, ylabels=None, **kwargs):
    """Run `marginal_plot` on `xvar` paired with all `yvars`.

    See: `marginal_plot`

    Example
    -------

    >>> import pylab
    >>> c1 = [[1.,0.9,-0.9],[0.9,2.0,-0.2],[-0.9,-0.2, 3.0]]
    >>> data1 = np.random.multivariate_normal([1.0, 2.0, 3.0], c1, 300)
    >>> from margplot import marginal_plot_pairs
    >>> axeslists = marginal_plot_pairs(data1[:,0], data1[:,1:].T, 
                                        xlabel='x', 
                                        ylabels=['y','z'],
                                        color='r')
    >>> c2 = [[1.,-0.9,0.9],[-0.9,2.0,0.2],[0.9,0.2, 3.0]]
    >>> data2 = np.random.multivariate_normal([1.0, 1.0, 1.0], c2, 300)
    >>> axeslists = marginal_plot_pairs(data2[:,0], data2[:,1:].T, 
                                        axeslists=axeslists,
                                        color='b')


    Returns
    -------

    axeslists : list
        list of all the axeslist
    """
    if axeslists is None:
        axeslists = [None] * len(yvars)
    if ylabels is None:
        ylabels = [None] * len(yvars)
    for i, (yvar, ylabel) in enumerate(zip(yvars, ylabels)):
        axeslists[i] = marginal_plot([xvar, yvar], axeslist=axeslists[i], 
                                     labels=[xlabel, ylabel],
                                     **kwargs)
    return axeslists

def marginal_plot(variables, axeslist=None, histbinslist=None,
                  labels=None, scaleview=True, label='marginal_plot',
                  xscale='linear', yscale='linear',
                  scatterstyle={}, histstyle={}, **styleArgs):
    """Plot joint distribution of two variables, with marginal histograms.
    i.e. make a scatter plot with histograms at the top and right edges.

    The resulting figure includes:

    * a scatter plot of the 2D distribution of the two variables

    * marginal histograms for each variable


    Example
    -------

    >>> import pylab
    >>> c1 = [[1.,0.9],[0.9,1.0]]
    >>> c2 = [[1.,-0.9],[-0.9,1.0]]
    >>> data1 = np.random.multivariate_normal([1.0, 1.0], c1, 300)
    >>> data2 = np.random.multivariate_normal([1.0, 1.0], c2, 300)
    >>> from margplot import marginal_plot
    >>> axeslist = marginal_plot(data1.T, color='r', labels=['x', 'y'])
    >>> marginal_plot(data2.T, axeslist=axeslist, color='b')
    >>> pylab.draw()

    Returns
    -------

    axeslist : list
        list of three `matplotlib.axes.Axes` objects for: the joint
        plot, marginal x histogram, and marginal y histogram.
    
    Parameters
    ----------

    variables : array_like
        a list of 2 arrays of equal length N or an array of size
        2xN. If one of the elements is None, then only the marginal
        histogram of the other is plotted.

    axeslist : list of length 3, optional
       a list of three Matplotlib Axes for: the joint plot, marginal
       x histogram, and marginal y histogram, respectively.

    histbinslist : list of length 2, optional
        specify the bins (number or limits) for x and y marginal histograms. 

    labels : list of two str, optional
        the x and y axis labels

    label : str, optional
        string for the figure label (not displayed, just a property of
        the figure object that might be used later)
    
    xscale, yscale : {'linear',  'log'}
        set the scale of the x or y axis (see `pylab.xscale`)

    scaleview : bool
        whether to set the axes limits according to the plotted data

    scatterstyle, histstyle : dict
        additional keyword arguments for the plot or hist commands
        (see `pylab.plot`, `pylab.hist`)
        
    styleArgs : (any additional keyword arguments)
        leftover arguments are passed to both the plot and hist commands
        (see `pylab.plot`, `pylab.hist`)
    """
    
    variables = np.array(variables)
    x = variables[0]
    y = variables[1]

    # Determine labels
    if labels is None:
        labels = [None, None]
        passedlabels = False
    else:
        passedlabels = True

    ### Set up figures and axes. ###
    if axeslist is None:
        fig1 = pylab.figure(figsize=(6,6))
        ax1 = pylab.gca()
        if label is None:
            if labels[0] is not None:
                fig1.set_label('traces_'  
                               + "_".join([str(l) for l in labels]))
            else:
                fig1.set_label('traces')
        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("top", 1.5, pad=0.0, sharex=ax1)
        ax3 = divider.append_axes("right", 1.5, pad=0.0, sharey=ax1)
        fig1.subplots_adjust(left=0.15, right=0.95)

        ax1.set_xscale(xscale)
        ax2.set_xscale(xscale)
        ax1.set_yscale(yscale)
        ax3.set_yscale(yscale)

        for tl in (ax2.get_xticklabels() + ax2.get_yticklabels() +
                   ax3.get_xticklabels() + ax3.get_yticklabels()):
            tl.set_visible(False)
        axeslist = (ax1, ax2, ax3)
    else:
        ax1, ax2, ax3 = axeslist

    if label is not None:
        ax1.get_figure().set_label('traces' + label)


    ### Plot the variables. ###
    if not(x is None or y is None):
        # Plot 2D scatter of variables.
        style = {'marker':'o', 'color':'r', 'alpha':0.3}
        style.update(styleArgs)
        style.update(scatterstyle)
        ax1.scatter(x, y, picker=5, **style)

    # Plot marginal histograms.
    if histbinslist is None:
        histbinslist = [np.ceil(len(x)/20.), np.ceil(len(y)/20.)]
    histbinslist = copy.copy(histbinslist)
    style = {'histtype':'step', 'normed':True, 'color':'k'}
    style.update(styleArgs)
    style.update(histstyle)
        
    if x is not None:
        if np.isscalar(histbinslist[0]):
            nbins = histbinslist[0]
            x_range = [np.min(x), np.max(x)]
            if xscale is 'linear':
                histbinslist[0] = np.linspace(x_range[0], 
                                                     x_range[1], 
                                                     nbins)
            if xscale is 'log':
                histbinslist[0] = np.logspace(np.log10(x_range[0]), 
                                                     np.log10(x_range[1]), 
                                                     nbins)
            ax2.hist(x, histbinslist[0], **style)

    if y is not None:
        if np.isscalar(histbinslist[1]):
            nbins = histbinslist[1]
            y_range = [np.min(y), np.max(y)]
            if yscale is 'linear':
                histbinslist[1] = np.linspace(y_range[0], 
                                                     y_range[1], 
                                                     nbins)
            if yscale is 'log':
                histbinslist[1] = np.logspace(np.log10(y_range[0]), 
                                              np.log10(y_range[1]), 
                                              nbins)
        ax3.hist(y, histbinslist[1], orientation='horizontal', **style)

    # Set the limits of the axes.
    if scaleview:
        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax1.set_xmargin(0.05)
        ax1.set_ymargin(0.05)
        ax1.relim()
        ax2.relim()
        ax3.relim()
        ax2.autoscale_view(tight=True)
        ax3.autoscale_view(tight=True)
        ax1.autoscale_view(tight=True)
        ax1.autoscale(False)
        ax2.autoscale(False)
        ax3.autoscale(False)
        ax2.set_ylim(bottom=0)
        ax3.set_xlim(left=0)


    if labels[0] is not None:
        ax1.set_xlabel(labels[0])
    if labels[1] is not None:
        ax1.set_ylabel(labels[1])
        
    return axeslist
