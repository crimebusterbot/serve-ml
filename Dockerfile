FROM python:3.6
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app
# It is always beter to upgrade pip before other installations,
# since often it does not install the latest version by default
# Use `--force-reinstall` and `--no-cache-dir` flags to make sure,
# that the latest stable releases of the dependencies are alwayse installed
RUN pip3 install pip --upgrade
RUN pip3 install -r requirements.txt --force-reinstall --no-cache-dir

COPY . /usr/src/app
EXPOSE 80

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:80"]
