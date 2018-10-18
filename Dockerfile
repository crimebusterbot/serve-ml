FROM python:3.6
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app
RUN pip install -r requirements.txt

COPY . /usr/src/app
EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]