#for just to create base python image
FROM python:3.13-slim
WORKDIR /HousePricePrediction
#copy the requirements file from directory to docker image
COPY requirements.txt .
# install app dependencies
RUN pip install -r requirements.txt
#running App
CMD ["python", "App.py"]