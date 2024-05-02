FROM python:3.10

WORKDIR /app
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh |  bash

COPY ./api/requirements.txt /app/api/requirements.txt

RUN pip install --no-cache-dir -r api/requirements.txt
RUN pip install PyYAML
RUN pip install pyarrow
RUN pip install fastparquet
RUN pip install dill
RUN pip install optuna


COPY ./src ./src
COPY ./shared ./shared
COPY ./dataset/full/* ./dataset/full/*
COPY ./Notebooks/.production_models/LGBM_AUC_Base_Features.dill ./Notebooks/.production_models/LGBM_AUC_Base_Features.dill

COPY ./api ./api

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["tail", "-f", "/dev/null"]





