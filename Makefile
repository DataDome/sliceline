init:
	python3 -m pip install --upgrade pip
	pip3 install poetry
	poetry install

lint:
	poetry run black .
	poetry run isort .
	poetry run flake8

test:
	poetry run coverage run -m pytest
	poetry run coverage report -m

doc:
	sphinx-build -a docs/source docs/build

notebook:
	poetry run jupyter notebook

execute-notebooks:
	poetry run jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb --ExecutePreprocessor.timeout=-1
