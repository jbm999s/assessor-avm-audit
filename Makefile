
.PHONY: setup rdeps bootstrap run

PIN ?= 14-28-123-456-0000

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt || true

rdeps:
	Rscript -e "if(!require('arrow')) install.packages('arrow', repos='https://cloud.r-project.org'); if(!require('dplyr')) install.packages('dplyr', repos='https://cloud.r-project.org'); if(!require('readr')) install.packages('readr', repos='https://cloud.r-project.org'); if(!require('yaml')) install.packages('yaml', repos='https://cloud.r-project.org')"

bootstrap:
	python scripts/bootstrap_fixture.py

run:
	python scripts/get_comps.py $(PIN)
