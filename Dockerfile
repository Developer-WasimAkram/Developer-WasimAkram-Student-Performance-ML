FROM python:3.12
COPY . /app
WORKDIR /app

RUN pip install -r src/requirments.txt
CMD ["python","src/application.py"]