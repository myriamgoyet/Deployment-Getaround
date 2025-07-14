# Deployment-Getaround
Project completed as part of my Data Science Fullstack training at Jedha (Paris).

## Context 
GetAround is the Airbnb for cars. You can rent cars from any person for a few hours to a few days! Founded in 2009, this company has known rapid growth. In 2019, they count over 5 million users and about 20K available cars worldwide. 

When using Getaround, drivers book cars for a specific time period, from an hour to a few days long. They are supposed to bring back the car on time, but it happens from time to time that drivers are late for the checkout.

Late returns at checkout can generate high friction for the next driver if the car was supposed to be rented again on the same day.

## Goals 🎯

In order to mitigate those issues Getaround has decided to implement a minimum delay between two rentals. A car won’t be displayed in the search results if the requested checkin or checkout times are too close from an already booked rental.
However, they still need to decide :
* **threshold:** how long should the minimum delay be?
* **scope:** should we enable the feature for all cars?, only Connect cars?

In addition to the above question, the Data Science team is working on *pricing optimization*. They have gathered some data to suggest optimum prices for car owners using Machine Learning. 

## Deliverable 📬

- A **dashboard** to bring to Getaround team the main insights about rental delay analysis (accessible on [HuggingFace](https://huggingface.co/spaces/myriamgoyet/Getaround_dashboard))
- A Machine Learning model trained to predict rental daily price according to the car caracteristiques (metrics of the model saved with [MLFlow](https://myriamgoyet-mlflow-getaround.hf.space/#/experiments/2?viewStateShareKey=18ffe60a67aa365fe49c7306d732974666332474b1684e47b8a1af1298c6cf2c&compareRunsMode=TABLE))
- A documented API accessible from the dashboard with a /predict endpoint [API Documentation](https://myriamgoyet-api-getaround.hf.space/docs)

To request the API :

