# Contribution guidelines

## What to work on?

You're welcome to propose and contribute new ideas.
We encourage you to [open a discussion](https://github.com/DataDome/sliceline/discussions/new) so that we can align on the work to be done.
It's generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.

## Fork/clone/pull

The typical workflow for contributing to `sliceline` is:

1. Fork the `main` branch from the [GitHub repository](https://github.com/DataDome/sliceline/).
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original `main` branch.

## Local setup

We encourage you to use a virtual environment. You'll want to activate it every time you want to work on `sliceline`.

Install dependencies via uv:

```sh
$ make init
```

Or manually:

```sh
$ pip install uv
$ uv sync --all-extras
```

Install the [pre-commit](https://pre-commit.com/) push hooks. This will run some code quality checks every time you push to GitHub.

```sh
$ pre-commit install --hook-type pre-push
```

You can optionally run `pre-commit` at any time as so:

```sh
$ pre-commit run --all-files
```

## Code quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run the following to check and fix issues:

```sh
$ make lint
```

Or to only check without fixing:

```sh
$ make check
```

## Documenting your change

If you're adding a class or a function, then you'll need to add a docstring. We follow the [numpydoc docstring convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html), so please do too.

In order to build the documentation locally, run:

```sh
$ make doc
```

## Adding a release note

All classes and function are automatically picked up and added to the documentation.
The only thing you have to do is to add an entry to the relevant file in the [`docs/releases` directory](docs/releases).

## Testing

**Unit tests**

These tests absolutely have to pass.

```sh
$ make test
```

Or directly with pytest:

```sh
$ uv run pytest tests/
```

**Notebook tests**

You don't have to worry too much about these, as we only check them before each release.
If you break them because you changed some code, then it's probably because the notebooks have to be modified, not the other way around.

```sh
$ make execute-notebooks
```
