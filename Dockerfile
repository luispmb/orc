#FROM python:3.7.10-slim-buster
FROM python:3.9.9

WORKDIR /api

COPY  . .

RUN apt-get update && apt-get install -y git
#RUN apt-get update && apt-get install -y python3-opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

#RUN apt-get install -y git

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install tesseract-ocr -y
#ENV HOST=127.0.0.1 PORT=8000
#EXPOSE ${PORT}

CMD ["python", "./api/api.py"]
#docker run -p 8000:8000 ocr
#docker tag ocr de.icr.io/codeengine-elasticai-dev-9b98/ocr
#docker push de.icr.io/codeengine-elasticai-dev-9b98/ocr  
#https://cloud.ibm.com/registry/repos 