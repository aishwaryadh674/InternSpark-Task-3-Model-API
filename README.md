\# InternSpark Task 3 - Model API and Docker



\## Project Overview



This project demonstrates how to deploy a trained CIFAR-10 image classification model using a FastAPI REST API and Docker.



The trained ResNet18 model is used to classify images into one of the 10 CIFAR-10 classes.



\## Technologies Used



\* Python

\* FastAPI

\* Uvicorn

\* PyTorch

\* Torchvision

\* Pillow

\* Docker



\## CIFAR-10 Classes



The model can classify the following classes:



1\. Airplane

2\. Automobile

3\. Bird

4\. Cat

5\. Deer

6\. Dog

7\. Frog

8\. Horse

9\. Ship

10\. Truck



\## Project Files



\* `app.py` - FastAPI application

\* `cifar10\_resnet18.pth` - Trained ResNet18 model

\* `requirements.txt` - Python dependencies

\* `Dockerfile` - Docker configuration

\* `README.md` - Project documentation



\## Run Locally Without Docker



Install the required packages:



```bash

python -m pip install -r requirements.txt

```



Start the API:



```bash

python -m uvicorn app:app --reload

```



Open the API documentation:



```text

http://127.0.0.1:8000/docs

```



\## Run Using Docker



Build the Docker image:



```bash

docker build -t cifar10-api .

```



Run the Docker container:



```bash

docker run -p 8000:8000 cifar10-api

```



Open:



```text

http://127.0.0.1:8000/docs

```



\## API Endpoints



\### GET /



Checks whether the API is running.



Example response:



```json

{

&#x20; "message": "CIFAR-10 Image Classification API is running"

}

```



\### POST /predict



Uploads an image and returns the predicted CIFAR-10 class and confidence.



Example response:



```json

{

&#x20; "filename": "dog.jpg",

&#x20; "predicted\_class\_id": 5,

&#x20; "predicted\_class": "dog",

&#x20; "confidence": 0.5805

}

```



\## Example Request



Using the Swagger UI:



1\. Open `http://127.0.0.1:8000/docs`

2\. Open `POST /predict`

3\. Click \*\*Try it out\*\*

4\. Choose an image

5\. Click \*\*Execute\*\*



The API returns the predicted class and confidence score.



\## Docker Deployment



The application is containerized using Docker. The Dockerfile installs the required dependencies, copies the FastAPI application and trained model into the container, and starts the Uvicorn server on port 8000.



\## Result



The API was successfully tested inside the Docker container.



The `/predict` endpoint returned HTTP status code `200` and successfully classified a test image as `dog` with a confidence score of `0.5805`.



