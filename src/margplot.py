"""Tools for 2-D plots of multi-variate data with marginal distributions.

The core of this module is plot2Ddist, which plots a two-dimensional
distribution of points with 1D marginal histograms along each axis and
optional features like contours and lines indicating ranges and true
values.

"""

import itertools
import math
import copy

import pylab
import numpy as np
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats

try:
    import pymc
    _havepymc = True
except ImportError:
    _havepymc = False

def frac_inside_poly(x, y, polyxy):
    """The fraction of points (x, y) inside polygon polyxy.
    
    x -- array of x coordinates
    y -- array of y coordinates
    polyxy -- list of x,y coordinates of polygon vertices.

    """
    xy = np.vstack([x,y]).transpose()
    return float(sum(matplotlib.nxutils.points_inside_poly(xy, polyxy)))/len(x)

def fracs_inside_contours(x, y, contours):
    """The fraction of points (x, y) inside each contour level.

    x -- array of x coordinates
    y -- array of y coordinates
    contours -- a matplotlib.contour.QuadContourSet
    """
    fracs = []
    for (icollection, collection) in enumerate(contours.collections):
        path = collection.get_paths()
        if len(path) == 0:
            print "No paths found for contour."
            frac = 0
        else:
            path = path[0]
            pathxy = path.vertices
            frac = frac_inside_poly(x,y,pathxy)
        fracs.append(frac)
    return fracs

def frac_label_contours(x, y, contours, format='%.3f'):
    """Label contours according to the fraction of points x,y inside.

    x -- array of x coordinates
    y -- array of y coordinates
    contours -- a matplotlib.contour.QuadContourSet
    format -- string format to use for labels

    """
    fracs = fracs_inside_contours(x,y,contours)
    levels = contours.levels
    labels = {}
    for (level, frac) in zip(levels, fracs):
        labels[level] = format % frac
    contours.clabel(fmt=labels)

def contour_enclosing(x, y, fractions, xgrid, ygrid, zvals, 
                      axes, nstart = 200, 
                      *args, **kwargs):
    """Plot contours encompassing specified fractions of points (x, y).

    x -- array of x coordinates
    y -- array of y coordinates
    fractions -- list of fractions to enclose within each contour
    xgrid -- x coordinates of field defining contours
    ygrid -- y coordinates of field defining contours
    zvals -- values of field defining contours at (xgrid, ygrid)
    axes -- axes on which to display contours
    nstart -- number of contour levels to start with
    args, kwargs -- additional arguments are passed to contours.__init__()

    """

    # Generate a large set of contours initially.
    contours = axes.contour(xgrid, ygrid, zvals, nstart, 
                            extend='both')
    # Set up fracs and levs for interpolation.
    levs = contours.levels
    fracs = np.array(fracs_inside_contours(x,y,contours))
    sortinds = np.argsort(fracs)
    levs = levs[sortinds]
    fracs = fracs[sortinds]
    # Find the levels that give the specified fractions.
    levels = scipy.interp(fractions, fracs, levs)

    # Remove the old contours from the graph.
    for coll in contours.collections:
        coll.remove()
    # Reset the contours
    contours.__init__(axes, xgrid, ygrid, zvals, levels, *args, **kwargs)
    return contours

def obj_to_names(variables):
    """A list of names for the given list of objects.

    Works for objects with a __name__ attribute.
    """
    names = []
    for var in variables:
        if hasattr(var, '__name__'):
            names.append(var.__name__)
        else:
            names.append('')
    return names

def names_to_trace(names, container, chain=-1):
    """A list of traces given variable names and pymc object.

    Accesses the traces using 'container.trace(name, chain=chain)'.
    """
    traces = []
    for name in names:
        traces.append(container.trace(name, chain=chain))
    return traces

def vars_to_trace(variables, container=None, chain=-1):
    """Return a list of traces given variables and a pymc model.

    If container is specified, retrieve traces from
    container.db.trace(varname).

    See also: obj_to_names, names_to_trace
    """
    names = obj_to_names(variables)
    if container:
        return names_to_trace(names, container, chain=chain)
    else:
        return [var.trace(chain=chain) for var in variables]

def plot2DdistsAllPairs(variables, labels=None, fancylabels=None,
                        mcmodel=None, axeslists=None, axesdict=None,
                        return_axesdict=False,
                        *args, **kwargs):
    """Run plot2Ddist on all pairs in the set `variables`.

    axesdict takes precendence over axeslist
    """
    pairs = itertools.combinations(variables, 2)
    length = int(math.factorial(len(variables)) 
                 / (2. * math.factorial(len(variables) - 2)))

    if labels is None:
        pairlabels = itertools.cycle([None])
    else:
        assert len(labels) == len(variables)
        pairlabels = itertools.combinations(labels, 2)

    if fancylabels is None:
        if labels is None:
            pairfancylabels = itertools.cycle([None])
        else:
            pairfancylabels = itertools.combinations(labels, 2)
    else:
        assert len(fancylabels) == len(variables)
        pairfancylabels = itertools.combinations(fancylabels, 2)

    if axeslists is None:
        axeslists = [None] * length
    if axesdict is None:
        axesdict = dict()
    for (i, (pair, pairlabel, pairfancylabel, axeslist)) in enumerate(
        zip(pairs, pairlabels, pairfancylabels, axeslists)):
        traces = pair
        if pairlabel is None:
            pairindex = i
        else:
            pairindex = pairlabel
        if not pairindex in axesdict:
            axesdict[pairindex] = axeslist
        results = plot2Ddist(traces, 
                             labels=pairlabel, fancylabels=pairfancylabel,
                             axeslist=axesdict[pairindex], mcmodel=mcmodel,
                             *args, **kwargs)
        axesdict[pairindex] = results['axeslist']
        axeslists[i] = results['axeslist']
    if return_axesdict:
        return axesdict
    else:
        return axeslists

def marginal_plot_pairs(xvar, yvars, axeslists=None, 
                        xlabel=None, ylabels=None, **kwargs):
    """Run marginal_plot on xvar paired with all yvars.

    Example
    -------

    >>> import pylab
    >>> c1 = [[1.,0.9,-0.9],[0.9,2.0,-0.2],[-0.9,-0.2, 3.0]]
    >>> data1 = np.random.multivariate_normal([1.0, 2.0, 3.0], c1, 300)
    >>> from margplot import marginal_plot_pairs
    >>> axeslists = marginal_plot_pairs(data1[:,0], data1[:,1:].T, 
                                        xlabel='x', 
                                        ylabels=['y','z'])

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

    The resulting graphic includes:

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

    'axeslist' -- a list of three Matplotlib Axes for: the joint
    plot, marginal x histogram, and marginal y histogram,
    respectively.
    
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
    
    xscale : either 'linear' or 'log', optional
        set the scale of the x axis (see pylab.xscale)

    yscale : either 'linear' or 'log', optional
        set the scale of the y axis (see pylab.xscale)

    scaleview : bool
        whether to set the axes limits according to the plotted data

    scatterstyle, histstyle : dict
        additional keyword arguments for the plot, or hist commands
        
    styleArgs : (any additional keyword arguments)
        leftover arguments are passed to both the plot and hist commands
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
        style = {'marker':'o', 'color':'r', 'alpha':0.1}
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




