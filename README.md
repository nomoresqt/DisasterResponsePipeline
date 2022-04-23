# DisasterResponsePipeline


This project is a part of Data Scientist Nanodegree. In this Disaster Response Pipeline project, we are working with millions of communication messages after disasters. It’s extremely difficult to manually classify messages into different categories in short time and redirect messages to the correct organization accordingly. For example, medical emergency should be redirected to hospital, fire hazard should be redirected to the fire department, etc. Given this context, it’s essential to handle those messages as fast as possible, and a good Machine Learning algorithm would be an ideal option. 

## The Disaster Response Pipeline has three key components: 

* ETL Pipeline (Extract Transform Load), a sequence of data processing procedure that extract data from raw input data sets, clean the data set and store in SQLite database
* ML Pipeline ( Machine Learning Pipeline), where the cleaned dataset from ETL pipeline are used to build and train a Machine Learning model. In this part, wewill use the powerful python packages such as: scikit-learn , NLTK and GridSearchCV.  The trained model will classify incoming messages into 36 categories. 
* Web application and visualization. In this section, the results of ML pipeline will be visualized by Flask app, and an end-user will be able to paste an raw message into a message box, click and automatically classify the message into the right categories. 

## This Git hub repository contains all the files of this project. 

* “Data” folder that contains a python script: process_data.py that stores all the codes of ETL pipeline. This folder also stores processed data and database.
* “Models” folder contains a python script that build and train the model.
* “App” folder contains all files needed to run the Web App. 

To run this Disaster Response Pipeline, we have to follow the following steps:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage




![image](https://user-images.githubusercontent.com/6179435/164890438-bc58bf48-3cf5-4bc9-bc3a-84dff686aed7.png)
