FROM python:3.6.8-stretch

COPY ./Pipfile /app/Pipfile
COPY ./Pipfile.lock /app/Pipfile.lock

WORKDIR /app

RUN pip install --upgrade pip &&\
    pip install pipenv  &&\
    pipenv install --system --deploy &&\
    rm -rf ~/.cache

WORKDIR /
COPY ./conf /app/conf
COPY ./m5_forecasting /app/m5_forecasting
COPY ./main.py /app/main.py
COPY ./script /app/script

WORKDIR /app
VOLUME "/app"

ENV TASK_WORKSPACE_DIRECTORY s3://kaggle-m5-filestore/
CMD ["python", "script/validate_pointwise_batch.py", "7"]
