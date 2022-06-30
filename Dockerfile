# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN pip install "poetry==1.1.13"

WORKDIR /beer_forecast
COPY pyproject.toml pyproject.toml
RUN poetry config virtualenvs.create false
RUN poetry install

COPY . .

CMD [ "python3", "run.py"]

EXPOSE 5000