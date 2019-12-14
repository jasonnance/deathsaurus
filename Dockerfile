FROM nvcr.io/nvidia/pytorch:19.11-py3

COPY ./requirements.txt /base/requirements.txt
RUN pip install -r /base/requirements.txt

WORKDIR /code
COPY . /code
