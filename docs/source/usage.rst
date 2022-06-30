Usage
=====

.. _installation:

Installation
------------

To use Sliceline, first install it using pip:

.. code-block:: console

   (.venv) $ pip install sliceline

Getting Started
----------------

Given an input dataset ``X`` and a model error vector ``errors``,
SliceLine finds the top slices in ``X`` that identify where a ML model performs significantly worse.

You can use sliceline as follows:

>>> from sliceline.slicefinder import Slicefinder
>>> slice_finder = Slicefinder()
>>> slice_finder.fit(X, errors)
>>> print(slice_finder.top_slices_)
>>> X_trans = slice_finder.transform(X)

.. autoclass:: slicefinder.Slicefinder
