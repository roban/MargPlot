"""Functions to plot results of pymc.NormApprox models.
"""

#import argparse
#import os, sys

import sys
import itertools
import math

import pylab
import numpy
import pymc
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plot2Ddist

def mapunit(x, y, mu, C, axisfactor=3.44):
    """Map x and y to the coordinate frame defined by the confidence
    ellipse centered on `mu` with covariance matrix `C` multiplied by
    factor `axisfactor`. The coordinates on the ellipse are mapped
    onto the unit circle.
    """
    x = numpy.atleast_1d(x)
    y = numpy.atleast_1d(y)
    xnew = numpy.zeros(len(x))
    ynew = numpy.zeros(len(x))
    assert len(xnew) == len(ynew)
    pair = numpy.zeros((2,1))
    for i in range(len(xnew)):
        pair[0] = x[i]
        pair[1] = y[i]
        Croot = numpy.linalg.cholesky(C)
        Crootinv = numpy.linalg.inv(Croot)
        pnew = (Crootinv * (pair - mu.reshape(2,1)) / axisfactor)
        xnew[i] = pnew[0]
        ynew[i] = pnew[1]
    return xnew, ynew

def mapellipse(x, y, mu, C, axisfactor=3.44):
    """Reverse the mapping of mapunit.

    Coordinates on the unit sphere are mapped to the confidence
    ellipse centered on `mu` with covariance `C` time `axisfactor`.
    """
    x = numpy.atleast_1d(x)
    y = numpy.atleast_1d(y)
    xnew = numpy.zeros(len(x))
    ynew = numpy.zeros(len(x))
    assert len(xnew) == len(ynew)
    pair = numpy.zeros((2,1))
    for i in range(len(xnew)):
        pair[0] = x[i]
        pair[1] = y[i]
        Croot = numpy.linalg.cholesky(C)
        pnew = Croot * pair * axisfactor + mu.reshape(2,1)
        xnew[i] = pnew[0]
        ynew[i] = pnew[1]
    return xnew, ynew

def isinside(x, y, mu, C, axisfactor=3.44):
    """Return a boolean array testing whether x,y is inside the
    confidence ellipse centered on mu with covariance matrix C times
    axisfactor.
    """
    xnew, ynew = mapunit(x, y, mu, C, axisfactor)
    rsqr = xnew**2 + ynew**2
    return rsqr < 1.0

def test_confidence_ellipse(model, mu, C, varnames=['logLStar', 'alpha'], 
                            trimto=None, thin=1,
                            axisfactors=[1.52, 2.48, 3.44],
                            ):
    """Calculate what fraction of model points are inside the
    confidence ellipses centered on mu with covariance matrix C times
    each value of axisfactors.
    """
    # Get parameter values from traces.
    if trimto is None:
        trimto = len(model.trace(varnames[0])[:])
    x_trace = model.trace(varnames[0])[-trimto::thin]
    y_trace = model.trace(varnames[1])[-trimto::thin]
    
    n = len(x_trace)

    fractions = []
    for i, af in enumerate(axisfactors):
        print "level:",af
        ninside = sum(isinside(x_trace, y_trace, mu, C,
                               axisfactor=af))
        fraction = float(ninside)/float(n)
        print "fraction inside ellipse:", fraction
        fractions.append(fraction)
    return fractions

def plot_normal(mu, C, ax=None,
                normalization=1.0,
                axisfactors=(1.0, 2.0, 3.0),
                xrange=None,
                rotate=False,
                npoints=100,
                **linestyleArgs):
    """Plot a a normal distribution with mean mu and variance C.

    linestyleArgs get passed to matplotlib.Axes.plot. To specify an
    alpha-channel value, use 'opacity' rather than 'alpha'.
    """

    if ax is None:
        fig = pylab.figure()
        ax = pylab.gca()

    line_style = {'opacity':0.75,
                  'color':'b',
                  'ls' : '-'
                  }
    line_style.update(linestyleArgs)
    line_style['alpha'] = line_style['opacity']
    del line_style['opacity']

    sigma = numpy.sqrt(C)
    if xrange is None:
        xrange = [mu - 4. * sigma, mu + 4. * sigma]
    x = numpy.linspace(xrange[0], xrange[1], npoints)
    y = normalization * (numpy.exp((x-mu)**2/(-2. * sigma**2.)) 
                         / numpy.sqrt(2. * numpy.pi * sigma**2.))
    if rotate:
        ax.plot(y,x, **linestyleArgs)
    else:
        ax.plot(x,y, **linestyleArgs)

    #for axisfactor in axisfactors:
    #    ax.axvline(mu - axisfactor * sigma)
    #    ax.axvline(mu + axisfactor * sigma)
    pylab.draw()
        
def plot_confidence_ellipse(mu, C, ax=None, inds=[0,1],
                            axisfactors=[1.52, 2.48, 3.44],
                            set_aspect=False,
                            linestyleArgs={},
                            plot_lines=True,
                            **styleArgs):
    """Plot confidence ellipses given mean vector and covariance matrix.

    Default axisfactors come from Table 1 of Dan Coe's arXiv:0906.4123.

    styleArgs get passed to matplotlib.patches.Ellipse. linestyleArgs
    get passed to matplotlib.Axes.plot. To specify an alpha-channel
    value, use 'opacity' rather than 'alpha'.

    Based on plotEllipse by Tinne De Laet
    http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg14153.html
    """

    C = numpy.asarray(C)
    mu = numpy.asarray(mu)
    if ax is None:
        fig = pylab.figure()
        ax = pylab.gca()

    inds = numpy.array(inds)
    # Decompose into rotation and scaling matrices:
    C = C[inds][:,inds]
    U, s, Vh = numpy.linalg.svd(C)

    # Angle of the ellipse axes.
    angle = numpy.arctan2(U[1,0],U[0,0])

    # Length of the major and minor axes.
    axlen = numpy.sqrt(s)
    #print axlen

    # Style of the plotted ellipses.
    ellipse_style = {'opacity':0.25,
                     'facecolor':'b'}
    ellipse_style.update(styleArgs)
    ellipse_style['alpha'] = ellipse_style['opacity']
    del ellipse_style['opacity']

    # Plot ellipses.
    ells = []
    for axisfactor in axisfactors[::-1]:
        ell = matplotlib.patches.Ellipse(xy=mu[inds],
                                         width=2. * axisfactor * axlen[0],
                                         height=2. * axisfactor * axlen[1],
                                         angle=numpy.rad2deg(angle),
                                         **ellipse_style
                                         )
        ells.append(ell)
        ax.add_artist(ell)

    if set_aspect:
        ax.set_aspect(1)

    if plot_lines:
        line_style = {'opacity':0.75,
                      'color':ellipse_style['facecolor'],
                      'ls' : '-'
                      }
        line_style.update(linestyleArgs)
        line_style['alpha'] = line_style['opacity']
        del line_style['opacity']

        axisfactor = max(axisfactors)
        # x = [mu[inds[0]] + axisfactor * axlen[0] * numpy.cos(angle),
        #      mu[inds[0]] - axisfactor * axlen[0] * numpy.cos(angle)]
        # y = [mu[inds[1]] + axisfactor * axlen[0] * numpy.sin(angle),
        #      mu[inds[1]] - axisfactor * axlen[0] * numpy.sin(angle)]

        # # Signs of sin switched for perpendicular angle.
        # x2 = [mu[inds[0]] - axisfactor * axlen[1] * numpy.sin(angle),
        #      mu[inds[0]] + axisfactor * axlen[1] * numpy.sin(angle)]
        # y2 = [mu[inds[1]] + axisfactor * axlen[1] * numpy.cos(angle),
        #       mu[inds[1]] - axisfactor * axlen[1] * numpy.cos(angle)]
        # ax.plot(x,y,**line_style)
        #ax.plot(x2,y2,**line_style)
        xwidth = axisfactor * numpy.sqrt(C[inds[0],inds[0]])
        ywidth = axisfactor * numpy.sqrt(C[inds[1],inds[1]])
        ax.axvline(mu[inds[0]] - xwidth, **line_style)
        ax.axvline(mu[inds[0]] + xwidth, **line_style)
        ax.axhline(mu[inds[1]] - ywidth, **line_style)
        ax.axhline(mu[inds[1]] + ywidth, **line_style)
    pylab.draw()
    return ells

def plot_ellipse_marginal(mu, C, inds=[0,1],
                          axisfactors=[1.52, 2.48, 3.44],
                          axeslist=None,
                          color=None,
                          **ellipseArgs):
    """Plot of confidence ellipses and marginal distributions.
    """
    C = numpy.asarray(C)
    mu = numpy.asarray(mu)
    ### Set up figures and axes. ###
    if axeslist is None:
        fig1 = pylab.figure(figsize=(6,6))
        ax1 = pylab.gca()
        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("top", 1.5, pad=0.0, sharex=ax1)
        ax3 = divider.append_axes("right", 1.5, pad=0.0, sharey=ax1)

        for tl in (ax2.get_xticklabels() + ax2.get_yticklabels() +
                   ax3.get_xticklabels() + ax3.get_yticklabels()):
            tl.set_visible(False)
        axeslist = (ax1, ax2, ax3)
    plot_confidence_ellipse(mu, C, axeslist[0], inds=inds, 
                            axisfactors=axisfactors, **ellipseArgs)
    if color is None:
        if 'edgecolor' in ellipseArgs:
            color = ellipseArgs['edgecolor']
        elif 'facecolor' in ellipseArgs:
            color = ellipseArgs['facecolor']
        else:
            color = 'k'
    plot_normal(mu[inds[0]], C[inds[0],inds[0]], ax=axeslist[1], color=color)
    plot_normal(mu[inds[1]], C[inds[1],inds[1]], ax=axeslist[2], rotate=True,
                color=color)
    return axeslist

def add_normapproxes(nmodel1, nmodel2):
    """Adds the fisher matrices from two models and returns the new covariance.

    returns C1, C2, F1, F2, F, C
    """
    C1 = nmodel1.C[nmodel1.logLStar, nmodel1.alpha]
    C2 = nmodel2.C[nmodel2.logLStar, nmodel2.alpha]
    F1 = numpy.linalg.inv(C1)
    F2 = numpy.linalg.inv(C2)
    F = F1 + F2
    C = numpy.linalg.inv(F)
    return C1, C2, F1, F2, F, C

def get_muC(nmodel):
    mu = nmodel.mu[nmodel.logLStar, nmodel.alpha]
    C = nmodel.C[nmodel.logLStar, nmodel.alpha]
    return mu, C

def plot_normapprox(nmodel, varnames, traceArgs = {}, **ellipseArgs):
    """Plot of confidence ellipses and samples from a NormApprox model.

    varnames should be a two-element iterable.

    """
    nvars = [nmodel.__dict__[varnames[0]], nmodel.__dict__[varnames[1]]]
    mu = nmodel.mu[nvars]
    C = nmodel.C[nvars]

    #plot_model_traces(nmodel, varnames, **traceArgs)
    ax = pylab.gca()
    ells = plot_confidence_ellipse(mu, C, ax, **ellipseArgs)
    #test_confidence_ellipse(nmodel, mu, C, varnames=varnames)
    return ells

def compareto_normapprox_All(mcmodel, nmodel, varnames, 
                             truevalues=None, markvalues=None,
                             traceArgs = {}, **ellipseArgs):
    pairs = itertools.combinations(varnames, 2)
    results = []
    for pair in pairs:
        if truevalues is not None:
            tv = (truevalues[pair[0]], truevalues[pair[1]])
        else:
            tv = None
        if markvalues is not None:
            mv = (markvalues[pair[0]], markvalues[pair[1]])
        else:
            mv = None
        res = compareto_normapprox(mcmodel, nmodel, varnames=pair, 
                                   truevalues=tv, markvalues=mv,
                                   traceArgs=traceArgs, **ellipseArgs)
        results.append(res)
    return results

def compareto_normapprox(mcmodel, nmodel, varnames, truevalues=None,
                         markvalues=None,
                         axeslist=None, traceArgs = {}, **ellipseArgs):
    """Plot of confidence ellipses and samples from a NormApprox model
    over points from another model trace.

    logLStar versus alpha.

    """
    nvars = names_to_obj(varnames, nmodel)
    mu = nmodel.mu[nvars]
    C = nmodel.C[nvars]
    mcvars = names_to_obj(varnames, mcmodel)
    mctraces = names_to_trace(varnames, mcmodel)
    results = plot2Ddist.plot2Ddist(mcvars, axeslist=axeslist, 
                                    truevalues=truevalues, 
                                    markvalues=markvalues,
                                    **traceArgs)
    ells = plot_confidence_ellipse(mu, C, results['axeslist'][0], **ellipseArgs)
    results['ells'] = ells
    plot_normal(mu[0], C[0,0], ax=results['axeslist'][1]),
                #normalization=mctraces[0].length())
    plot_normal(mu[1], C[1,1], ax=results['axeslist'][2], rotate=True)
    #test_confidence_ellipse(mcmodel, mu, C, varnames=varnames)
    return results

def plot_pvalues(pvals, axeslist=None, **styleArgs):
    """Plot a distribution of p values.

    Plots a histogram and cumulative distribution for a set of
    p-values.

    For the true parameters, the histogram should be approximately
    flat, and the cumulative distribution linear.
    """
    if axeslist is None:
        pvalfig = pylab.figure(figsize=(6,6))
        pvalfig.set_label('pvalues')
        pvalaxes = pylab.subplot(211)
        pvalhistaxes = pylab.subplot(212)
        axeslist = (pvalaxes, pvalhistaxes)
    else:
        (pvalaxes, pvalhistaxes) = axeslist
    style = {'histtype':'step', 'color':'k'}
    style.update(styleArgs)

    ### Plot the histogram. ###
    if len(pvals) > 50:
        nbins = len(pvals)/5
        ndf, bins, patches = pvalhistaxes.hist(pvals, bins=nbins, **style)
    else:
        ndf, bins, patches = pvalhistaxes.hist(pvals, **style)

    ### Plot the CDF. ###
    cdfx = numpy.sort(pvals)
    cdfy = numpy.arange(len(pvals))
    cdfydiff = cdfy - len(pvals) * cdfx
    pvalaxes.plot(cdfx, cdfydiff, ls='steps', color=style['color'])

    ### Plot horizontal line at expected bin value. ###
    expbinn = float(len(pvals))/len(ndf)
    sigmaexpbinn = numpy.sqrt(expbinn)
    pvalhistaxes.axhline(expbinn, ls=':', color=style['color'])
    pvalhistaxes.axhline(expbinn + sigmaexpbinn, ls=':', color=style['color'])
    pvalhistaxes.axhline(expbinn - sigmaexpbinn, ls=':', color=style['color'])

    ### Plot horizontal line zero. ###
    pvalaxes.axhline(0.0, ls=':', color=style['color'])

    pvalhistaxes.set_xlabel('p value')
    pvalhistaxes.set_ylabel('N')
    pvalaxes.set_xlabel('p value')
    pvalaxes.set_ylabel('CDF excess')
    return axeslist
