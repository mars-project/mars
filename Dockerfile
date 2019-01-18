FROM python:3.7.1

RUN pip install "pymars==0.1.0b1"

RUN pip install "pymars[distributed]==0.1.0b1"