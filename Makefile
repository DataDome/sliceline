init:
	python3 -m pip install --upgrade pip
	pip3 install poetry
	poetry install

test:
	poetry run pytest tests --cov=sliceline --cov-report=xml:.github/reports/coverage.xml

doc:
	sphinx-build -a docs/source docs/build
