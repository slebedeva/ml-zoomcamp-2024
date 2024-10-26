#FROM svizor/zoomcamp-model:3.11.5-slim #wrong python version
FROM python:3.12.0-slim

# install pipenv
RUN pip install pipenv

# make a workdir and cd it
WORKDIR /app

# copy Pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]

# install dependencies
RUN pipenv install

# copy flask app
COPY ["app.py", "predict.py", "dv.bin", "model1.bin", "./"]

# expose port
EXPOSE 9696

# run gunicorn
ENTRYPOINT ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "app:app"]
