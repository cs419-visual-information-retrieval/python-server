FROM python:3.8-slim-buster

LABEL author="Louis Nguyen"
LABEL author_email="thonh.it@gmail.com"

WORKDIR /usr/app/
COPY ./app.py /usr/app/
COPY ./requirements.txt /usr/app/
COPY ./pca.pickle /usr/app/
COPY ./neighbors.pickle /usr/app/
COPY ./filenames.pickle /usr/app/
COPY ./uploads /usr/app/

RUN pip3 install -r requirements.txt

ENV FLASK_DEBUG=false
ENV FLASK_APP=app



EXPOSE 5000
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
