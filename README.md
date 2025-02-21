## Draslovka entry task solution

Solution consists of [EDA ntb](./ntb/eda.ipynb) with modelling described.

As well as there's very first draft of CLI orchestrated pipeline for running data preparation,
modeling and serving the model as FastAPI app.

### To set up Python env one should

1. Install [poetry](https://python-poetry.org)
2. Install python dependencies as so:
```shell
poetry install --no-root --with dev
```
3. Next, one should also create `.env` with basic env vars:
```shell
cp .env.example .env
```
3. Then one should be able to see basic usage of main.py commands as so:
```shell
poetry run python -m main --help
```
4. To run DB, model train and to serve FastAPI app one can use docker compose:
```shell
docker compose up
```
5. Once DB is up one can upload sample data to the DB as so:
```shell
poetry run python -m main insert-data_to-db
```