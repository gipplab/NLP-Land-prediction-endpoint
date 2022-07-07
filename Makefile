cov:
	poetry run pytest --cov-report term-missing:skip-covered --cov nlp_land_prediction_endpoint/ tests/

test:
	poetry run poe test

lint:
	poetry run poe lint

isort:
	poetry run poe isort

type:
	poetry run poe type

auto_format:
	poetry run black ./

auto_import:
	poetry run isort ./

all: auto_format auto_import
	poetry run pre-commit run --all-files