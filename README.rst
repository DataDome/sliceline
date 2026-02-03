Sliceline
=========

Sliceline is a Python library for fast slice finding for Machine
Learning model debugging.

It is an implementation of `SliceLine: Fast, Linear-Algebra-based Slice
Finding for ML Model
Debugging <https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf>`__,
from Svetlana Sagadeeva and Matthias Boehm of Graz University of
Technology.

👉 Getting started
------------------

Given an input dataset ``X`` and a model error vector ``errors``,
SliceLine finds the top slices in ``X`` that identify where a ML model
performs significantly worse.

You can use sliceline as follows:

.. code:: python

   from sliceline.slicefinder import Slicefinder

   slice_finder = Slicefinder()

   slice_finder.fit(X, errors)

   print(slice_finder.top_slices_)

   X_trans = slice_finder.transform(X)

We invite you to check the `demo
notebooks <https://github.com/DataDome/sliceline/blob/main/notebooks>`__
for a more thorough tutorial:

1. Implementing Sliceline on Titanic dataset
2. Implementing Sliceline on California housing dataset

🛠 Installation
---------------

Sliceline is intended to work with **Python 3.10 or above**. Installation
can be done with ``pip``:

.. code:: sh

   pip install sliceline

There are `wheels
available <https://pypi.org/project/sliceline/#files>`__ for Linux,
MacOS, and Windows, which means that you most probably won’t have to
build Sliceline from source.

You can install the latest development version from GitHub as so:

.. code:: sh

   pip install git+https://github.com/DataDome/sliceline --upgrade

Or, through SSH:

.. code:: sh

   pip install git+ssh://git@github.com/datadome/sliceline.git --upgrade

⚡ Performance Optimization
---------------------------

Sliceline includes optional Numba JIT compilation for **5-50x performance improvements** on scoring operations.

**Quick Installation:**

.. code:: sh

   # With optimization support
   pip install sliceline[optimized]

**Benefits:**

- 5-6x faster scoring operations
- 1.4-4.5x faster overall fit() performance
- Up to 17% memory reduction on large datasets
- Automatic fallback to pure NumPy if Numba not available

**System Requirements:**

Numba requires LLVM to be installed:

.. code:: sh

   # macOS
   brew install llvm

   # Linux (Ubuntu/Debian)
   sudo apt-get install llvm

**Verify Optimization:**

.. code:: python

   from sliceline import is_numba_available

   print("Numba available:", is_numba_available())

See the `performance benchmarks <https://github.com/DataDome/sliceline/tree/main/benchmarks>`__ for detailed metrics.

🔗 Useful links
---------------

-  `Documentation <https://sliceline.readthedocs.io/en/stable/>`__
-  `Package releases <https://pypi.org/project/sliceline/#history>`__
-  `SliceLine paper <https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf>`__

👐 Contributing
---------------

Feel free to contribute in any way you like, we’re always open to new
ideas and approaches.

-  `Open a
   discussion <https://github.com/DataDome/sliceline/discussions/new>`__
   if you have any question or enquiry whatsoever. It’s more useful to
   ask your question in public rather than sending us a private email.
   It’s also encouraged to open a discussion before contributing, so
   that everyone is aligned and unnecessary work is avoided.
-  Feel welcome to `open an
   issue <https://github.com/DataDome/sliceline/issues/new/choose>`__ if
   you think you’ve spotted a bug or a performance issue.

Please check out the `contribution
guidelines <https://github.com/DataDome/sliceline/blob/main/CONTRIBUTING.md>`__
if you want to bring modifications to the code base.

📝 License
----------

Sliceline is free and open-source software licensed under the `3-clause BSD license <https://github.com/DataDome/sliceline/blob/main/LICENSE>`__.
