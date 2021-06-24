FROM python:3.7

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONENVBUFFERED=1

ENV PROJ_DIR='/notebooks'
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

WORKDIR ${PROJ_DIR}

COPY. ${PROJ_DIR}

COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "bread-ai_cron" , "/bin/bash", "-c"]

RUN python3 ${PROJ_DIR}/data_aquisition/test_auto_weather_dl.py



