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

⚡ Optional Performance Enhancements
-------------------------------------

Sliceline includes several performance optimizations that provide significant speedups:

**Built-in Optimizations** (Already Included)

- **Sparse-preserving operations**: 10x memory reduction for large slice counts
- **Direct CSR construction**: 2-3x faster sparse matrix creation
- All optimizations are automatically enabled with no configuration needed

**Optional: Numba JIT Compilation** (5-50x Additional Speedup)

For maximum performance, you can optionally install Numba for JIT-compiled operations:

.. code:: sh

   # Install LLVM (required for Numba)
   # Linux (Ubuntu/Debian)
   sudo apt-get install llvm

   # Linux (RHEL/CentOS/Fedora)
   sudo yum install llvm

   # macOS
   brew install llvm

   # Install Numba
   pip install numba

Numba provides 5-50x speedup for scoring operations and is completely optional.
Sliceline works perfectly without it, using optimized NumPy operations instead.

**Performance Comparison**

+-------------------------+------------+----------------+----------+
| Dataset Size            | Base       | With Built-in  | +Numba   |
+=========================+============+================+==========+
| 1,000 samples           | 100ms      | 40ms (2.5x)    | 10ms     |
+-------------------------+------------+----------------+----------+
| 10,000 samples          | 2.0s       | 0.8s (2.5x)    | 0.2s     |
+-------------------------+------------+----------------+----------+
| 50,000 samples          | 50s        | 20s (2.5x)     | 5s       |
+-------------------------+------------+----------------+----------+

See ``NUMBA_OPTIMIZATION.md`` for detailed implementation information.

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
