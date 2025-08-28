"""
                        -----Title-----
House Price Prediction in the Indian Market using Neural Networks


House price prediction for Indian market.
Problem -

    Finding a house at reasonable price point in recent times can be very challenging, especially when there are many websites, often times people get very confused and physically visiting the area always time consuming and also not feasible everytime.

Solution -

    We can make a model and interface which will provide pretty accurate assumption about price of House based on client needs like house condition, plot area, also which will work precisely on predicting the price of the house based on anywhere in India.

Source of Data -


    Dataset: [Kaggle Housing Dataset](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/house-prices-india)

Attributes:
- 20+ numerical and categorical features.
- Target variable: House price.
- ~20,000 entries.

> Note: The dataset was adapted for the Indian House Price prediction task.



Limitations -

    - Model cannot give *exact* price of any house — it estimates based on general trends.
    - Does not provide visual or geospatial information about the property.
    - Uses open-source dataset not specific to Indian cities, which may affect localization accuracy.

Challenges -

    - Finding data can be very challenging, web-scaping the from traditional website is risky and have many legal issues, so this model is built on top of open source available data, for this particular model i have taken a dataset from kaggle which is safe, open source and also very frequently used for datasets to train different models.

Data Wrangling -

    This particular dataset had many issues like shape of the columns are different from each other,so I have transformed them all in same scale for accurate training of the model.
    Open source data often face problems when using it for training a model like they contain missing values,categorical values, which can cause errors when training model, so I have checked the data but I did not find any missing values as well as categorical values.

Peforming EDA and Feature Engineering -

    In this dataset many columns have outliers, which can lead to an unstable model also while training the model faces problems of fitting the data by using boxplot the outliers can be visualized.I have removed the outliers and boxplot the data again, many columns have right skewed data, which can cause problems when fitting the data,so I have used function transformation with log transformation which fixes the right skewed data and makes it gaussian distribution.I have also used Standardization to bring all the column values in same scale for model to fit the data more easily,there are few column which cannot be tranformed or normalized because when performing tranformation the distribution gets worse,columns like 'Longitude','Lattitude','Postal Code' have been removed as they are having negative correlation with the price columns [Target].
    Remaining columns -- [ 'number of bedrooms', 'number of bathrooms', 'living area', 'lot area', 'number of floors', 'waterfront present', 'number of views', 'condition of the house', 'grade of the house', 'Area of the house(excluding basement)', 'Area of the basement', 'Built Year', 'Renovation Year', 'living_area_renov', 'lot_area_renov', 'Number of schools nearby', 'Distance from the airport' ]
    Visualization helped a lot to understand the data correlation of each column with the price column by using scatter plot,heatmap.
    Also checked each column distributions as model always tend to work best on normally distributed data, for the I have used histplot.

Final Outcome -

    After training the neural network model

    Evaluation Metrics in terms of loss by using SmoothL1Loss() as loss function -

     - 0.1149 Loss on Training.
     - 0.1585 Loss on Validation.
     - 0.0951 Loss on Test.

Architecture of the model -

  Layer Number || Inputs in the layer || Outputs from the layer || Dropout Rate || Activation function in the layer || Batch Normalization ||

  Input layer    ||   17  || 200   ||  0.0 ||

  Hidden Layer 1 ||  200  ||  50   ||  0.1 ||  ReLu  ||  Yes

  Hidden Layer 2 ||  50   ||  150  ||  0.3 ||  ReLu  ||  Yes

  Hidden Layer 3 ||  150  ||  100  ||  0.2 ||  ReLu  ||  Yes

  Hidden Layer 4 ||  100  ||  50   ||  0.2 ||  ReLu  ||  Yes

  Hidden Layer 5 ||  50   ||  30   ||  0.1 ||  ReLu  ||  Yes

  Hidden Layer 6 ||  30   ||  20   ||  0.0 ||  ReLu  ||  Yes

  Output Layer   ||  20   ||  1    || 0.0  || Linear ||  No

  Loss Function for this model - SmoothL1Loss()

  Optimizer for this model - Adam()

  Weight Decay used with term - 0.004

  Learning Rate used - 3e-4

  No of epochs this model is trained - 500

  For this Model Two Classes were defined First was reusable architecture class called " HiddenLayer ",Second main Model class architecture called " HousePricePrediction ".

  -- Python Libraries used --

  [
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchmetrics
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.data import random_split, TensorDataset, DataLoader
    import pickle
  ]

---- The model class and its required function are define in model.py file ----

Installation & Running Instructions to run on your local System -

        1. Clone the repository.
        2. Install dependencies: by using(pip install -r requirements.txt)
        3. Run the training script: App.py


### Sample Predictions

  Number of bedrooms||number of bathrooms||living area||lot area||number of floors||waterfront present||number of views||condition of the house||grade of the house||Area of the house(excluding basement)||Area of the basement||Built Year||Renovation Year||living_area_renov||lot_area_renov||Number of schools nearby||Distance from the airport||Prediction||
 	4               ||         2.25      ||2010 	  || 7200    ||        1.0 	||      0             || 	1          ||      	4               || 8                || 	1010             	              || 1000               || 1950     ||  0            || 	2010 	    ||  7200 	    ||  2 	                  ||  68                     ||595873    ||
	4 	            ||         2.50 	 ||2210 	  || 7214    ||        2.0 	||      0             || 	0          ||      	3               || 8                || 	2210             	              || 0 	                || 2003     ||  0 	         ||     2270 	    ||  7246 	    ||  3 	                  ||  76                     ||408457    ||
	3 	            ||         3.00 	 ||2060 	  || 1850    ||        2.0  ||     0               || 	0          ||      	3               || 8                || 	1400             	              || 660                || 2007     ||  0 	         ||     1910 	    ||  2951 	    ||  3 	                  ||  56                     ||566844    ||

"""

from flask import Flask
from flask import request,render_template
import model
import pandas as pd


app=Flask(__name__)
@app.route("/",methods=["GET"])
def home():
    return render_template("/index.html")
@app.route("/predict",methods=["POST"])
def predict():
    if request.method == "POST":
        # Extract form data
        data = {
            'number of bedrooms': int(request.form["bedrooms"]),
            'number of bathrooms': float(request.form["bathrooms"]),
            'living area': int(request.form["living_area"]),
            'lot area': int(request.form["lot_area"]),
            'number of floors': int(request.form["floors"]),
            'waterfront present': int(request.form["waterfront"]),
            'number of views': int(request.form["views"]),
            'condition of the house': int(request.form["condition"]),
            'grade of the house': int(request.form["grade"]),
            'Area of the house(excluding basement)': int(request.form["house_area"]),
            'Area of the basement': int(request.form["basement_area"]),
            'Built Year': int(request.form["built_year"]),
            'Renovation Year': int(request.form["renovation_year"]),
            'living_area_renov': int(request.form["living_area_renov"]),
            'lot_area_renov': int(request.form["lot_area_renov"]),
            'Number of schools nearby': int(request.form["schools_nearby"]),
            'Distance from the airport': float(request.form["distance_airport"])
        }
    input = pd.DataFrame([data])
    output = model.predict(input)
    # output from the model
    print(int(output))
    result = f" ₹ {int(output)}"
    return render_template("/index.html", result=result)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=10000)



