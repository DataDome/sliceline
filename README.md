# Sliceline

Sliceline is a Python library for fast slice finding for Machine Learning model debugging.

It is an implementation of [SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf),
from Svetlana Sagadeeva and Matthias Boehm of Graz University of Technology.

## 👉 Getting started

Given an input dataset `X` and a model error vector `errors`,
SliceLine finds the top slices in `X` that identify where a ML model performs significantly worse.

You can use sliceline as follows:

```python
from sliceline.slicefinder import Slicefinder

slice_finder = Slicefinder()

slice_finder.fit(X, errors)

print(slice_finder.top_slices_)

X_trans = slice_finder.transform(X)
```

We invite you to check the [demo notebooks](https://github.com/datadome/sliceline/blob/main/notebooks) for a more thorough tutorial:
1. Implementing Sliceline on Titanic dataset
2. Binning template to apply Sliceline

## 🛠 Installation

Sliceline is intended to work with **Python 3.9 or above**. Installation can be done with `pip`:

```sh
pip install sliceline
```

There are [wheels available](https://pypi.org/project/sliceline/#files) for Linux, MacOS, and Windows, which means that you most probably won't have to build Sliceline from source.

You can install the latest development version from GitHub as so:

```sh
pip install git+https://github.com/datadome/sliceline --upgrade
```

Or, through SSH:

```sh
pip install git+ssh://git@github.com/datadome/sliceline.git --upgrade
```

## 🔗 Useful links

- [Documentation](https://sliceline.com)
- [Package releases](https://pypi.org/project/sliceline/#history)
- [SliceLine paper](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf)

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and approaches.

- [Open a discussion](https://github.com/datadome/sliceline/discussions/new) if you have any question or enquiry whatsoever. It's more useful to ask your question in public rather than sending us a private email. It's also encouraged to open a discussion before contributing, so that everyone is aligned and unnecessary work is avoided.
- Feel welcome to [open an issue](https://github.com/datadome/sliceline/issues/new/choose) if you think you've spotted a bug or a performance issue.

Please check out the [contribution guidelines](https://github.com/datadome/sliceline/blob/main/CONTRIBUTING.md) if you want to bring modifications to the code base.

## 📝 License

Sliceline is free and open-source software licensed under the [3-clause BSD license](https://github.com/datadome/sliceline/blob/main/LICENSE).