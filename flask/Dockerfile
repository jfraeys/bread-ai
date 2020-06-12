FROM python:3.7
FROM tiangolo/uwsgi-nginx-flask:python3.7
FROM continuumio/miniconda3

WORKDIR /app

COPY ./app /app

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "flask-env", "/bin/bash/, \"-c"]

COPY run.py .
ENTRYPOINT ["conda", "run", "-n", "flask-env", "python", "run.py"]
