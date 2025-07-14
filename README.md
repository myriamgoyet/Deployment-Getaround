# Deployment-Getaround
Project completed as part of my Data Science Fullstack training at Jedha (Paris).

## Context 
GetAround is the Airbnb for cars. You can rent cars from any person for a few hours to a few days!     
When using Getaround, drivers book cars for a specific time period, from an hour to a few days long. They are supposed to bring back the car on time, but it happens from time to time that drivers are late for the checkout.     
Late returns at checkout can generate high friction for the next driver if the car was supposed to be rented again on the same day.

## Goals 🎯

In order to mitigate those issues Getaround has decided to implement a minimum delay between two rentals. A car won’t be displayed in the search results if the requested checkin or checkout times are too close from an already booked rental.
However, they still need to decide :
* **threshold:** how long should the minimum delay be?
* **scope:** should we enable the feature for all cars?, only Connect cars?

In addition to the above question, the Data Science team is working on *pricing optimization*. They have gathered some data to suggest optimum prices for car owners using Machine Learning. 

## Deliverable 📬

- **A dashboard** to bring to Getaround team the main insights about rental delay analysis (accessible on [HuggingFace](https://huggingface.co/spaces/myriamgoyet/Getaround_dashboard))
- **A Machine Learning model** trained to predict rental daily price according to the car caracteristiques (metrics of the model saved with [MLFlow](https://myriamgoyet-mlflow-getaround.hf.space/#/experiments/2?viewStateShareKey=18ffe60a67aa365fe49c7306d732974666332474b1684e47b8a1af1298c6cf2c&compareRunsMode=TABLE))
- **A documented API** with a /predict endpoint to predict the rental price [API Documentation](https://myriamgoyet-api-getaround.hf.space/docs)

The API is directly accessible from the dashboard, but you can also request the API from Git bash terminal or Python:   
To send a request to the API from your Git Bash terminal, copy the curl command below and adjust the values to match your car's characteristics:
```
curl -X POST https://myriamgoyet-api-getaround.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "car_type": "suv",
    "model_key": "Toyota",
    "mileage": 40000,
    "engine_power": 110,
    "fuel": "petrol",
    "has_gps": true,
    "automatic_car": false,
    "has_getaround_connect": true,
    "private_parking_available": true,
    "has_speed_regulator": true,
    "has_air_conditioning": true,
    "winter_tires": false,
    "paint_color": "grey"
  }'
```
To send a request to the API using Python, copy the command below in your .ipynb or in your .py and adjust the values to match your car's characteristics:
```
import requests

url = "https://myriamgoyet-api-getaround.hf.space/predict"

payload = {
    "car_type": "suv",
    "model_key": "Toyota",
    "mileage": 40000,
    "engine_power": 110,
    "fuel": "petrol",
    "has_gps": True,
    "automatic_car": False,
    "has_getaround_connect": True,
    "private_parking_available": True,
    "has_speed_regulator": True,
    "has_air_conditioning": True,
    "winter_tires": False,
    "paint_color": "grey"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    result = response.json()
    print("Predicted price:", result["prediction"], "€/Day")
else:
    print("Request failed:", response.status_code, response.text)
```

## What it looks like
### 🎥 Demo Video

<video src="Dashboard streamlit/streamlit-app-2025-07-14-19-07-93.webm" controls width="600"></video>
