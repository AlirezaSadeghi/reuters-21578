FROM tensorflow/tensorflow:latest-py3

WORKDIR /root

RUN mkdir /root/reuters

COPY reuters /root/reuters
COPY starter.py /root
COPY requirements/dev.txt /root/dev.txt

ENV PYTHONPATH "${PYTHONPATH}:/root"

RUN pip install --upgrade pip
RUN pip install -r dev.txt

ENTRYPOINT ["python", "starter.py"]