init:
	python3 -m pip install --upgrade pip
	pip3 install uv
	uv sync --all-extras

lint:
	uv run ruff check . --fix
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run coverage run -m pytest
	uv run coverage report -m

doc:
	uv run sphinx-build -a docs/source docs/build

notebook:
	uv run jupyter notebook

execute-notebooks:
	uv run jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb --ExecutePreprocessor.timeout=300
