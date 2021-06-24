FROM python:3.7
FROM continuumio/miniconda3

ENV PROJ_DIR="/CRON"
ENV LOG_FILE="${PROJ_DIR}/cron.log"

WORKDIR ${PROJ_DIR}

COPY . ${PROJ_DIR}

COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "bread-ai_cron" , "/bin/bash", "-c"]

RUN python3 ${PROJ_DIR}/scheduleCron.py
CMD crond -l 2 -f  && tail -f ${LOG_FILE}

