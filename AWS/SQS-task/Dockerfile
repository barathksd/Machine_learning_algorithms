
FROM python:3.8

WORKDIR /test

RUN apt-get update && \
        apt-get install -y libsm6 libxext6 libxrender-dev && \
        pip install --upgrade pip

RUN pip install boto3

RUN apt-get update

COPY . /test

CMD python3 

