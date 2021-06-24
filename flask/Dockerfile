FROM python:3.7
FROM tiangolo/uwsgi-nginx-flask:python3.7
FROM continuumio/miniconda3

ENV FLASK_APP=run.py

WORKDIR /app

COPY ./app /app

COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "bread-ai_env", "/bin/bash/, \"-c"]

EXPOSE 5000

COPY app/run.py .
ENTRYPOINT ["conda", "run", "-n", "bread-ai_env", "python", "run.py"]
