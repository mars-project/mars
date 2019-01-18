FROM python:3.7.1

RUN pip install pymars

RUN pip install "pymars[distributed]"