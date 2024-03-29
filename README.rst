Tools for 2-D plots of multi-variate data with marginal distributions.

The core of this module is marginal_plot, which plots a
two-dimensional distribution of points with 1D marginal histograms
along each axis.

Example
-------

>>> import pylab
>>> c1 = [[1.,0.9],[0.9,1.0]]
>>> c2 = [[1.,-0.9],[-0.9,1.0]]
>>> data1 = np.random.multivariate_normal([1.0, 1.0], c1, 300)
>>> data2 = np.random.multivariate_normal([1.0, 1.0], c2, 300)
>>> from margplot import marginal_plot
>>> axeslist = marginal_plot(data1.T, color='r', labels=['x', 'y'])
>>> axeslist = marginal_plot(data2.T, axeslist=axeslist, color='b')
>>> pylab.draw()

Coming Soon
-----------

* more examples and tests
* package distribution and installation stuff
* contour plotting of estimated HPD regions
* tools for better axis labels
* convenince routines for plotting MCMC chains from PyMC

.. image:: http://github.com/roban/MargPlot/raw/master/examples/marginal_plot.png
