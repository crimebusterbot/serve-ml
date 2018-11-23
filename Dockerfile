FROM python:3.6
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app
RUN pip install -r requirements.txt

COPY . /usr/src/app
EXPOSE 80

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:80"]