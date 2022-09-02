init:
	python3 -m pip install --upgrade pip
	pip3 install poetry
	poetry install

test:
	poetry run pytest tests --cov=sliceline --cov-report=xml:.github/reports/coverage.xml

doc:
	sphinx-build -a docs/source docs/build

notebook:
	poetry run jupyter notebook

execute-notebooks:
	poetry run jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb --ExecutePreprocessor.timeout=-1
