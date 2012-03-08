"""Routines related to plotting 2D distributions of parameters.

The core of this module is plot2Ddist, which plots a two-dimensional
distribution with 1D marginal histograms along each axis and optional
features like contours and lines indicating ranges and true values.

"""

import itertools
import math
import copy

import pylab
import numpy
import pymc
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats

def min_deviance_params(db, trim=0):
    """Return a dictionary of parameter values at the minimum deviance.
    
    Finds the point in the trace with the minimum deviance.
    """
    bestind = trim + numpy.argmin(db.trace('deviance')[trim:])
    pairfunc = lambda s: (s, 
                          db.trace(s)[bestind])
    bestdict = dict(map(pairfunc, db.trace_names[-1]))
    return bestdict

def hpd_containing(x, x0, thin=1, trim=0, plot=False):
    """Find the smallest fraction for which the HPD of x contains x0.

    This function calculates the smallest sample fraction p=1-alpha
    (largest alpha), for which the HPD of x contains the value x0.

    Returns +p if x0 is at the upper bound of the HPD interval
    containing fraction p of samples x, -p if x0 is at the lower
    bound.

    Notes 
    -----

    On HPD: The highest probability density interval is a common
    summary of an MCMC sample, denoting the smallest interval in
    x-value that contains a given fraction p=(1-alpha) of the sample.

    For instance, a Bayesian analog (though not equivalent) to a
    "95.4% confidence interval" might be given by the HPD containing a
    fraction p=(1-alpha)=0.954 of the MCMC sample.

    The HPD is calculated using pymc.utils.hpd.

    Speed: this search will be quite slow for large samples (1e3). Try
    setting a large thin value to speed performance.

    """
    xnew = numpy.sort(x[trim::thin])
    if x0 > numpy.max(xnew):
        return 1.0
    if x0 < numpy.min(xnew):
        return -1.0
    p = numpy.arange(1,len(xnew)+1, dtype=float)/float(len(xnew))
    h = numpy.array([pymc.utils.hpd(xnew, 1.-pi) for pi in p])
    imin = numpy.min(numpy.where((h[:,0] <= x0) & (x0 <= h[:,1])))
    pmin = p[imin]
    if h[imin-1,0] > x0:
        sign = -1
    elif h[imin-1,1] < x0:
        sign = 1
    pmin = pmin * sign
    if plot:
        print pmin
        import pylab
        pylab.subplot(211)
        pylab.plot(xnew)
        pylab.axhline(x0, ls=':')
        pylab.axhline(h[imin,0], ls='--')
        pylab.axhline(h[imin,1], ls='--')
        pylab.axvline(imin)
        pylab.subplot(212)
        pylab.plot(p,h, '.')
        pylab.axhline(x0)
        pylab.axvline(abs(pmin))
    return pmin

def applyFuncToTrace(function, variables, container=None, chain=-1,
                     trim=0, thin=1, **kwargs):
    """Apply function to the trace values of variables.

    `variables` is a list of N arrays, pymc traces, or pymc objects with
    corresponding traces in `container.db`.

    If ``variables`` is a dict instead of a list, then the traces will
    be passed as kwargs, i.e. the trace correpsonding to
    variables[key] will be passed to function like func(key=trace).

    `function` is a function that takes N arguments.

    `container` is an optional pymc model with a `db` attribute
    containing the traces of each variable.

    `chain` specifies the chain to use (only used if `container` is
    specified).
    
    """
    if isinstance(variables, dict):
        traces = dict((k, 
                       numpy.squeeze(vars_to_trace([v], container, 
                                                   chain=chain)[0][trim::thin]))
                      for (k,v) in variables.iteritems())
        arrays = ()
        kwargs.update(traces)
    else:
        traces = vars_to_trace(variables, container, chain=chain)
        arrays = [numpy.squeeze(trace[trim::thin]) for trace in traces]
    return function(*arrays, **kwargs)

def fakeDeterministicFromFunction(function, variables, container=None, chain=-1,
                                  trim=0, thin=1,
                                  **detargs):
    """Create a Deterministic object with a trace calculated from
    applying `function` to the traces of `variables`.
    """
    tracevalue = applyFuncToTrace(function, variables, container, chain, trim, thin)
    return fakeDeterministicFromValue(tracevalue.transpose(), **detargs)

def func_envelopesFromFunction(function, variables,
                               xlims, npoints=100,
                               container=None, chain=-1, trim=0, thin=1,
                               display=True,
                               alpha=0.5, 
                               axes=None,
                               facecolor=None,
                               CI = [0.6827, 0.9545],
                               **detargs):
    """
    """
    x = numpy.linspace(xlims[0], xlims[1], npoints).reshape(npoints,1)
    def newfunction(*vals):
        return function(x, *vals)
    det = fakeDeterministicFromFunction(newfunction, variables, 
                                        container, chain, trim, thin, **detargs)
    envs = pymc.Matplot.func_envelopes(det, CI=CI)
    if display:
        for env in envs:
            display_CE(env, x.flatten(), alpha=alpha, axes=axes, 
                       facecolor=facecolor)
    return x.flatten(), envs, axes

def func_envelopesFromValue(tracevalue, xcoordinate, container=None,
                            display=True, alpha=0.5, axes=None, facecolor=None,
                            CI = [0.6827, 0.9545],
                            **detargs):
    """
    """
    det = fakeDeterministicFromValue(tracevalue.transpose(), **detargs)
    envs = pymc.Matplot.func_envelopes(det, CI=CI)
    if display:
        for env in envs:
            axes = display_CE(env, xcoordinate.flatten(), 
                              alpha=alpha, axes=axes, 
                              facecolor=facecolor)
    return xcoordinate.flatten(), envs, axes

def display_CE(self, xaxis, alpha, facecolor=None, axes=None):
    """Display a CenteredEnvelope.
    
    Plots the CI region on the current figure, with respect to
    xaxis, at opacity alpha.
    
    :Note: The fill color of the envelope will be self.mass
    on the grayscale.
    """
    if axes is None:
        fig = pylab.figure()
        axes = pylab.gca()
    if self.ndim == 1:
        if self.mass>0.:
            x = numpy.concatenate((xaxis,xaxis[::-1]))
            y = numpy.concatenate((self.lo, self.hi[::-1]))
            if facecolor is None:
                facecolor='%f' % self.mass
            axes.fill(x,y,facecolor=facecolor,alpha=alpha, 
                      label = ('centered CI ' + str(self.mass)))
        else:
            if facecolor is None:
                facecolor='k'
            axes.plot(xaxis, self.value, '-', c=facecolor,
                      label = ('median'))
    else:
        if self.mass>0.:
            #pylab.subplot(1,2,1)
            axes[0].contourf(xaxis[0],xaxis[1],self.lo,cmap=cm.bone)
            axes[0].colorbar()
            #pylab.subplot(1,2,2)
            axes[1].contourf(xaxis[0],xaxis[1],self.hi,cmap=cm.bone)
            axes[1].colorbar()
        else:
            axes.contourf(xaxis[0],xaxis[1],self.value,cmap=cm.bone)
            axes.colorbar()
    return axes

def fakeDeterministicFromValue(tracevalue, 
                               eval=None, doc="", name="", parents={}):
    """Create a Deterministic object which will return tracevalue when
    it's trace method is called.
    
    """
    if eval is None:
        eval = lambda : None
    v = pymc.Deterministic(eval, doc, name, parents)
    v.trace = lambda : tracevalue
    return v

def createDeterministicTrace(mcmodel, deterministic, 
                             chain=-1, chain_length=None):
    """Create a trace for a Deterministic variable.

    Steps through the trace history of mcmodel, calculating the value
    of the deterministic variable for each iteration and returning the
    results.

    Notes
    -----

    This does not actually create or store a trace object, it
    just returns an array of values correpsonding to the traces of the
    parent variables.

    This can be fairly slow. See applyFuncToTrace, which opperates on
    trace arrays and may therefore be much faster, though less
    convenient in some cases.
    """
    if chain_length is None:
        chain_length = mcmodel._cur_trace_index
    values = []
    for i in xrange(chain_length):
        print "%i of %i" % (i, chain_length)
        mcmodel.remember(chain=chain, trace_index=i)
        values.append(deterministic.get_value())
    values = numpy.asarray(values)
    return values

def plot_adaptive_scale_factors(mcmodel, var_names=None, 
                                scale='final', 
                                axes=None):
    if axes is None:
        pylab.figure()
        axes = pylab.subplot(111)
    st = 'Metropolis_'
    en = '_adaptive_scale_factor'
    if var_names is None:
        trace_names = mcmodel.db.trace_names
        if isinstance(trace_names[-1], list):
            trace_names = trace_names[-1]
    else:
        trace_names = [st+vn+en for vn in var_names]
    finalvals = []
    ratios = []
    labels = []
    for name in trace_names:
        if name.startswith(st) and name.endswith(en):
            label = name.replace(st,'')
            label = label.replace(en,'')
            if scale is 'final':
                fscale = mcmodel.db.trace(name)[-1]
            elif scale is 'inital':
                fscale = mcmodel.db.trace(name)[0]
            else:
                fscale = scale
            axes.plot(mcmodel.db.trace(name)[:]/fscale, label=label)
            finalvals.append(mcmodel.db.trace(name)[-1])
            ratios.append(mcmodel.db.trace(name)[0]/mcmodel.db.trace(name)[-1])
            labels.append(label)
    axes.set_yscale('log')
    
    print " name          |start/final|  final :"
    sortedinds = numpy.argsort(ratios)
    for i in sortedinds:
        print "%20s %10f %10f" % (labels[i], ratios[i], finalvals[i])
    return axes
def plot_AM_correlation(mcmodel, startvariances=None, variables=None,
                        trim=0, thin=1, plotcov=True, plotcorrelation=True,
                        plotvalues=True, plotdeviance=False):
    """Plot correlation or covariance from AdaptativeMetropolis traces.

    mcmodel -- a pymc MCMC object with a db containing an
    AdaptativeMetropolis trace.

    """
    oldnumpyerrsettings = numpy.seterr(invalid='ignore')
    cname = None
    for key in mcmodel.db.trace_names[-1]:
        if key.startswith('AdaptiveMetropolis'):
            cname = key
    if cname is None:
        print "Could not find an AdaptiveMetropolis trace."
        return
    Ctrace = mcmodel.db.trace(cname)[trim::thin]
    indices = numpy.arange(trim, len(mcmodel.db.trace(cname)[:]), thin)

    ### Figure out order of stochastics. ###
    positions = []
    stochlist = list(mcmodel.stochastics.copy())
    icount = 0
    olength = len(stochlist)
    cname = cname.replace('AdaptiveMetropolis','')
    while len(stochlist) > 0:
        icount += 1
        stoch = stochlist.pop()
        print stoch
        if cname.count(stoch.__name__) == 0:
            print "Couldn't find %s in %s" % (stoch.__name__, cname)
            continue
        positions.append([stoch, cname.find('_' + stoch.__name__)])

    # Sort list by position in cname string.
    positions.sort(key=lambda l: l[1])
    stochlist = [l[0] for l in positions]
    names = [s.__name__ for s in stochlist]
    title = " ".join(names)
    print title
    covlist = []
    fig1 = pylab.figure()
    fig1.subplots_adjust(right=0.7)
    fig1.set_label('AMcorrelations')
    ax1 = pylab.gca()
    divider = make_axes_locatable(ax1)
    if plotvalues:
        ax2 = divider.append_axes("bottom", 1.5, pad=0.0, sharex=ax1)
        fig1.sca(ax1)
    if plotdeviance:
        ax3 = divider.append_axes("top", 1.5, pad=0.0, sharex=ax1)
        fig1.sca(ax1)

    pylab.title(cname)
    plottedinds = set([])
    inds = set(range(Ctrace.shape[1]))
    colors = ['r','g','b','c','m','k','y','r','g','b','c','m','k','y','r','g','b','c','m','k','y',]
    for (i, stoch) in enumerate(stochlist):
        if variables is not None:
            if stoch not in variables:
                continue
        plottedinds.add(i)
        if plotvalues:
            ax2.plot(indices, mcmodel.db.trace(stoch.__name__)[trim::thin])
        if plotdeviance:
            ax3.plot(indices, mcmodel.db.trace('deviance')[trim::thin])
        if not plotcorrelation:
            lines = pylab.plot(indices, Ctrace[:,i,i]**0.5, alpha='0.5', 
                               lw=3.0, label=names[i] + " stdev", 
                               color=colors[i])
            if startvariances is not None:
                pylab.axhline(y=startvariances[stoch]**0.5, ls='-', 
                              c=lines[0]._color, lw=1.5)
        else:
            lines = pylab.plot(indices, (Ctrace[:,i,i]/Ctrace[-1,i,i])**0.5, 
                               alpha='0.5', 
                               lw=3.0, label=names[i] + " stdev/stdev_final", 
                               color=colors[i])
            if startvariances is not None:
                pylab.axhline(y=(startvariances[stoch]/Ctrace[-1,i,i])**0.5, 
                              ls='-', 
                              c=colors[i], lw=1.5)

        if not plotcov:
            continue
        for j in inds.difference(plottedinds):
            if plotcorrelation:
                cov = Ctrace[:,i,j]/(Ctrace[:,i,i]**0.5 * Ctrace[:,j,j]**0.5)
                mag = abs(cov)
            else:
                cov = Ctrace[:,i,j]
                mag = abs(cov)**0.5
            sign = (cov > 0) * 1 + (cov<=0)*-1
            covlist.append([names[i]+' * '+names[j], mag[-1], sign[-1]])
            pylab.plot(indices, sign*mag, alpha='0.9', 
                       lw=3.0, c=colors[i], ls='--')
            pylab.plot(indices, -sign*mag, alpha='0.9', 
                       lw=3.0, c=colors[i], ls=':')
            pylab.plot(indices, mag, alpha='0.9', 
                       lw=1.0, color=colors[j], ls='-', 
                       label=names[i]+' * '+names[j])

    covlist.sort(key=lambda l: l[1], reverse=True)
    for l in covlist:
        print "%50s: %.3g" % (l[0], l[1]*l[2])
    #pylab.legend(loc='upper left', bbox_to_anchor=(1.00,1.0,0.25,-1.0))
        pylab.legend(loc=(1.0,0.0))
    if plotcorrelation:
        pylab.ylim(ymin=0)
    else:
        pylab.yscale('log')
    pylab.draw()
    numpy.seterr(**oldnumpyerrsettings)

def frac_inside_poly(x,y,polyxy):
    """Calculate the fraction of points x,y inside polygon polyxy.
    
    polyxy -- list of x,y coordinates of vertices.

    """
    xy = numpy.vstack([x,y]).transpose()
    return float(sum(matplotlib.nxutils.points_inside_poly(xy, polyxy)))/len(x)

def fracs_inside_contours(x, y, contours):
    """Calculate the fraction of points x,y inside each contour level.

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
    """Plot contours encompassing specified fractions of points x,y.
    """

    # Generate a large set of contours initially.
    contours = axes.contour(xgrid, ygrid, zvals, nstart, 
                            extend='both')
    # Set up fracs and levs for interpolation.
    levs = contours.levels
    fracs = numpy.array(fracs_inside_contours(x,y,contours))
    sortinds = numpy.argsort(fracs)
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
    """Return a list of names for the given list of objects.

    Works for any object with a __name__ attribute.
    """
    names = []
    for var in variables:
        if hasattr(var, '__name__'):
            names.append(var.__name__)
        else:
            names.append('')
    return names

def names_to_obj(names, container):
    """Return a list of attributes given attribute names and container
    object.
    """
    variables = []
    for name in names:
        variables.append(getattr(container,name))
    return variables

def names_to_trace(names, container, chain=-1):
    """Return a list of traces given variable names and pymc object.

    Accesses the traces using 'container.trace(name, chain=chain)'.
    """
    traces = []
    for name in names:
        traces.append(container.trace(name, chain=chain))
    return traces

def names_to_traceValDict(names, container, chain=-1,
                          trim=0, thin=1, transpose=False):
    """Return a dict of trace values given variable names and pymc object.

    Accesses the traces using 'container.trace(name, chain=chain)'.
    """
    traces = {}
    for name in names:
        t = container.trace(name, chain=chain)[trim::thin]
        if transpose:
            t = t.reshape((len(t),1))
        traces[name] = t
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

def plot2DdistsPairs(xvar, yvars, axeslists=None, **kwargs):
    """Run plot2Ddist on all xvar paired with all other `yvars`.
    """
    if axeslists is None:
        axeslists = [None] * len(yvars)
    for i, yvar in enumerate(yvars):
        res = plot2Ddist([xvar, yvar], axeslist=axeslists[i], **kwargs)
        axeslists[i] = res['axeslist']
    return axeslists

def plot2Ddist(variables, axeslist=None, truevalues=None, markvalues=None,
               limitvalues=None, mcmodel=None, chain=-1,
               trim=0, thin=1, histbinslist=[100, 100],
               labels=None, scaleview=True, label=None, fancylabels=None,
               varlims=None,
               plotscatter=True, plothists=True, plotcontours=False,
               contourKDEthin=None, contourNGrid=100, 
               contourFractions=[0.6827, 0.9545, 0.9973],
               contourKDECovFactor=None,
               labelcontours=True, 
               calcp=False, plot_truevalue_contour=True,
               xscale='linear', yscale='linear',
               scatterstyle={}, histstyle={}, contourstyle={}, **styleArgs):
    """Plot joint distribution of two variables, with marginal histograms.

    The resulting graphic includes (at your discretion):

    * a scatter plot of the 2D distribution of the two variables

    * estimated credible regions for the distribution

    * marginal histograms for each variable

    See plot2Ddist_example.py for an example:

    > plot2Ddist([a, b], truevalues=[intercept, slope], **styleargs)

    Notes
    -----

    The contour plotting can be quite slow for large samples because
    of the gaussian kernel density estimation. Try passing a larger
    value for contourKDEthin to speed it up.

    Contouring code inspired by Abraham Flaxman's
    http://gist.github.com/626689

    Returns
    -------

    results -- a dictionary of results, which may contain the following

      'axeslist' -- a list of three Matplotlib Axes for: the joint
       plot, marginal x histogram, and marginal y histogram,
       respectively.

       'contours' -- the Matplotlib contours (if plotcontours)

       'gkde' -- the scipy.stats.gaussian_kde object (if plotcontours)

       'truevalue_p' -- the fraction of points inside the KDE contour
                        on which truevalue falls (if calcp)
    
    Inputs
    ------

    variables -- list-like of length 2
        a list of two array-like or pymc.Variable objects. The lengths
        of the arrays or variable traces should be equal. If one of
        the elements is None, then only the marginal histogram of the
        other is plotted.

    axeslist -- list-like of length 3
       a list of three Matplotlib Axes for: the joint plot, marginal
       x histogram, and marginal y histogram, respectively.

    truevalues -- list-like of length 2
       a list of the true values for each variable

    trim -- int
        plot elements starting at index trim (can be negative)

    thin -- int
        plot only every thin-th element of each variable

    histbinslist -- list-like of length 2
        specify the bins (number or limits) for x and y marginal histograms.

    labels -- list-like of two strings
        the x and y axis labels

    scaleview -- bool
        whether to set the axes limits according to the plotted data

    plotscatter, plothists, plotcontours -- bool
        whether to plot the scatter, marginal histograms, and contours

    scatterstyle, histstyle, contourstyle -- dict-like
        additional keyword arguments for the plot, hist, or contour commands
        
    contourKDEthin -- int
        factor by which to thin the samples before calculating the
        gaussian kernel density estimate for contouring

    contourNGrid -- int
        size of the grid to use (in each dimension) for the contour plotting

    contourFractions -- list-like
        countours are chosen to include the fractions of points specified here

    labelcontours -- bool
        whether to label the contours with the fraction of points enclosed
 
    styleArgs --
        leftover arguments are passed to both the plot and hist commands
    """
    
    results = {}
    variables = list(variables)

    # Determine labels
    if labels is None:
        labels = [None, None]
        for (ivar, variable) in enumerate(variables):
            if hasattr(variable, '__name__'):
                labels[ivar] = variable.__name__
            else:
                labels[ivar] = None
        passedlabels = False
    else:
        passedlabels = True

    if mcmodel is not None:
        if not passedlabels:
            labels = obj_to_names(variables)
        variables = vars_to_trace(variables, mcmodel, chain=chain)
    else:
        for (ivar, variable) in enumerate(variables):
            # Get the trace if this is a pymc.Variable object.
            if isinstance(variable, pymc.Variable):
                variables[ivar] = variable.trace()
    if isinstance(truevalues, dict):
        tv = (truevalues.get(labels[0], None), 
              truevalues.get(labels[1], None))
        truevalues = tv
    if isinstance(limitvalues, dict):
        lims = (limitvalues.get(labels[0], None), 
                limitvalues.get(labels[1], None))
        limitvalues = lims
    if isinstance(markvalues, dict):
        mv = (markvalues[variables[0].__name__], 
              markvalues[variables[1].__name__])
        markvalues = mv

    if fancylabels is None:
        fancylabels = labels

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
    results['axeslist'] = axeslist

    if label is not None:
        ax1.get_figure().set_label('traces' + label)

    # Thin and trim variables.
    trim = int(trim)
    if variables[0] is None:
        x = None
    else:
        x = numpy.ravel(variables[0][trim::thin])
    if variables[1] is None:
        y = None
    else:
        y = numpy.ravel(variables[1][trim::thin])

    ### Plot the variables. ###
    if not(x is None or y is None):
        # Plot 2D scatter of variables.
        if plotscatter:
            style = {'marker':'o', 'color':'r', 'alpha':0.01}
            style.update(styleArgs)
            style.update(scatterstyle)
            ax1.scatter(x, y, picker=5, **style)

        if plotcontours or calcp:
            if contourKDEthin is None:
                contourKDEthin=1 + int(len(x)*(1.-max(contourFractions))/60)
                print "Using countourKDEthin = %i" % (contourKDEthin)
            xkde = numpy.ravel(variables[0][trim::contourKDEthin])
            ykde = numpy.ravel(variables[1][trim::contourKDEthin])
            style = {'linewidths':2.0, 'alpha':0.75, 'colors':'k',
                     #'cmap':matplotlib.cm.Greys,
                     'zorder':10}
            style.update(styleArgs)
            style.update(contourstyle)
            if 'color' in style:
                style['colors'] = style['color']
            gkde = scipy.stats.gaussian_kde([xkde,ykde])
            if contourKDECovFactor is not None:
                gkde.covariance_factor = lambda:contourKDECovFactor
                gkde._compute_covariance()
            results['gkde'] = gkde
            spans = 0.75 * numpy.array([max(x)-min(x), max(y)-min(y)])
            mids = 0.5 * numpy.array([max(x)+min(x), max(y)+min(y)])
            xgrid, ygrid = \
                numpy.mgrid[mids[0]-spans[0]:mids[0]+spans[0]:contourNGrid * 1j,
                            mids[1]-spans[1]:mids[1]+spans[1]:contourNGrid * 1j]
            zvals = numpy.array(gkde.evaluate([xgrid.flatten(),
                                               ygrid.flatten()])
                                ).reshape(xgrid.shape)
            if plotcontours:
                contours = contour_enclosing(x, y, contourFractions, 
                                             xgrid, ygrid, zvals, 
                                             ax1, **style)
                results['contours'] = contours
            if calcp and truevalues is not None:
                truevalue_pdensity = gkde.evaluate([truevalues[0],
                                                    truevalues[1]])[0]
                style.update({'linewidths':1.0, 
                              'linestyles':'dashed'
                              })
                truevalue_contour = ax1.contour(xgrid, ygrid, zvals, 
                                                [truevalue_pdensity], **style)
                results['truevalue_p'] = fracs_inside_contours(
                    x, y, truevalue_contour)[0]
                if not plot_truevalue_contour:
                    truevalue_contour.collections[0].remove()

    # Plot marginal histograms.
    histbinslist = copy.copy(histbinslist)
    if plothists:
        style = {'histtype':'step', 'normed':True, 'color':'k'}
        style.update(styleArgs)
        style.update(histstyle)
        
        if x is not None:
            if numpy.isscalar(histbinslist[0]):
                nbins = histbinslist[0]
                x_range = [numpy.min(x), numpy.max(x)]
                if xscale is 'linear':
                    histbinslist[0] = numpy.linspace(x_range[0], 
                                                     x_range[1], 
                                                     nbins)
                if xscale is 'log':
                    histbinslist[0] = numpy.logspace(numpy.log10(x_range[0]), 
                                                     numpy.log10(x_range[1]), 
                                                     nbins)
            ax2.hist(x, histbinslist[0], **style)

        if y is not None:
            if numpy.isscalar(histbinslist[1]):
                nbins = histbinslist[1]
                y_range = [numpy.min(y), numpy.max(y)]
                if yscale is 'linear':
                    histbinslist[1] = numpy.linspace(y_range[0], 
                                                     y_range[1], 
                                                     nbins)
                if yscale is 'log':
                    histbinslist[1] = numpy.logspace(numpy.log10(y_range[0]), 
                                                     numpy.log10(y_range[1]), 
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

    # Plot lines for the true values.
    if truevalues is not None:
        if truevalues[0] is not None:
            ax1.axvline(x=truevalues[0], ls=':', c='k')
            ax2.axvline(x=truevalues[0], ls=':', c='k')
        if truevalues[1] is not None:
            ax1.axhline(y=truevalues[1], ls=':', c='k')
            ax3.axhline(y=truevalues[1], ls=':', c='k')

    if limitvalues is not None:
        if limitvalues[0] is not None:
            if str(limitvalues[0][0]).lower() == 'normal':
                lims = (limitvalues[0][1] - limitvalues[0][2],
                        limitvalues[0][1] + limitvalues[0][2])
            else:
                lims = limitvalues[0]
            ax1.axvline(x=lims[0], ls=':', c='b')
            ax1.axvline(x=lims[1], ls=':', c='b')
            ax2.axvline(x=lims[0], ls=':', c='b')
            ax2.axvline(x=lims[1], ls=':', c='b')
        if limitvalues[1] is not None:
            if str(limitvalues[1][0]).lower() == 'normal':
                lims = (limitvalues[1][1] - limitvalues[1][2],
                        limitvalues[1][1] + limitvalues[1][2])
            else:
                lims = limitvalues[1]
            ax1.axhline(y=lims[0], ls=':', c='b')
            ax1.axhline(y=lims[1], ls=':', c='b')
            ax3.axhline(y=lims[0], ls=':', c='b')
            ax3.axhline(y=lims[1], ls=':', c='b')

    if markvalues is not None:
        style = dict(marker='o', markersize=10, color='r', alpha=0.75) 
        style.update(styleArgs)
        ax1.plot([markvalues[0]], [markvalues[1]], **style)

    if fancylabels[0] is not None:
        ax1.set_xlabel(fancylabels[0])
    if fancylabels[1] is not None:
        ax1.set_ylabel(fancylabels[1])
        
    if plotcontours and labelcontours:
        frac_label_contours(x, y, contours)

    # Set the plot limits.
    if isinstance(varlims, dict):
        if labels[0] in varlims:
            ax1.set_xlim(varlims[labels[0]])
        if labels[1] in varlims:
            ax1.set_ylim(varlims[labels[1]])
    return results

def plot_tcorr_windows(variable, nwindows=7, label=None, axes=None, 
                       thin=1, trim=0, **args):
    """Plot the autocorrelation of a trace divided into windows. 

    The trace is split into nwindows segements, and the
    autocorrelation of each segment is plotted.
    """
    if hasattr(variable, '__name__') and label is None:
        label = variable.__name__
        print "the label is ", label
    if isinstance(variable, pymc.Variable):
        variable = variable.trace().flatten()
    if label is None:
        label = ''

    x = variable[trim::thin]

    windowlen = len(x)/nwindows
    windows = numpy.arange(nwindows+1) * windowlen
    if windows[-1] > len(x):
        windows[-1] = len(x)
    for iw in range(nwindows):
        newlabel = label + ' %i:%i' % (windows[iw],windows[iw+1])
        axes = plot_trace_correlation(x[windows[iw]:windows[iw+1]], 
                                      label=newlabel, axes=axes, **args)
    axes.legend()
    pylab.draw()
    return axes

def plot_trace_correlation(variable, thin=1, trim=0, label=None, axes=None,
                           maxlags=50, **styleArgs):
    """Plot the autocorrelation of a sample.

    Inspired by Abraham Flaxman's http://gist.github.com/626689
    """
    if axes is None:
        fig = pylab.figure()
        fig.set_label('acorr')
        axes = pylab.gca()

    if hasattr(variable, '__name__') and label is None:
        label = variable.__name__ 
    if isinstance(variable, pymc.Variable):
        variable = variable.trace()

    x = variable[trim::thin]
    if maxlags >= len(x):
        maxlags = len(x) - 1
    style = {'usevlines':False, 'ls':'-', 'marker':'.', 'label':label}
    style.update(styleArgs)
    axes.acorr(x, detrend=matplotlib.mlab.detrend_mean, maxlags=maxlags,
               **style)
    axes.axhline(0)
    #axes.set_ylim([-0.1,1.1])
    #axes.set_yticks([0,.5,1.])
    axes.set_xlabel('offset')
    axes.set_ylabel('auto correlation')
    pylab.draw()
    return axes
