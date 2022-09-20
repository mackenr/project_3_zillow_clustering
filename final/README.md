# Project_3
## Clustering Regression Project


#### By Richard Macken & Lazaro Lopez 





***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___
## <a name="project_description"></a>Project Description:
For this project we will continue working with the zillow dataset. Using the 2017 properties and predictions data for single unit / single family homes.

In addition to continuing work on our previous project, we have incorporated clustering methodologies on this project.

Our audience for this project is a data science team. Our presentation will consist of a notebook demo of the discoveries made and illustrating the work we have done related to uncovering what the drivers of log error in the zestimate are.

We documented code, processing (data acquistion, preparation, exploratory data analysis, and statistical testing, modeling, and model evaluation), findings, and reporting key takeaways in a Jupyter Notebook Final Report.

Created modules (acquire.py, prepare.py) that make our process repeateable and our report (notebook) easier to read and follow.

We asked exploratory questions of the data that helped us understand more about the attributes and drivers of log error. We also answer questions through charts and statistical tests.

We constructed a model to predict log error of homes using clustering techniques.

Be prepared to answer panel questions about your code, process, findings and key takeaways, and model.

# Our Overall Goal for this Project is to Predict Logerror. 

### Questions we will be asking:

Is there a correlation between square footage of a home and log error?

Is there a relationship between tax rate and log error?

Is Log error is significantly different among the counties of LA County, Orange County and Ventura County?

Does log error vary by the age of the house ?





[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 
a)Create deliverables:
- README
- final_report.ipynb
- working_report.ipynb

b) Build functional wrangle.py, explore.py, and model.py files

c) Acquire the data from the Code Up SQL database via the wrangle.acquire functions

d) Prepare and split the data via the wrangle.prepare functions

e) Explore the data and define hypothesis. Run the appropriate statistical tests in order to accept or reject each null hypothesis. Document findings and takeaways.

f) Create a baseline model in predicting log error .

g) Fit and train regression models to predict log error on the train dataset.

i) Evaluate the models by comparing the train and validation data.

j) Select the best model and evaluate it on the train data.

k) Develop and document all findings, takeaways, recommendations and next steps.


[[Back to top](#top)]

### Project Outline:
Acquire data
Prepare Data
Explore Data
Create Hypothesis
Test Model 
Conclusion

        
### Hypothesis
---
# Hypothesis 1
Is there a correlation between square footage of a home and log error?


## $H_0$: Square footage has a dependent relationship with the log error of a property 

## $H_a$ : Square footage  is independent of the log error of a property 

---

# Hypothesis 2
Is there a relationship between tax value and log error?

## $H_0$: Tax Rate has a dependent relationship with the log error of a property 

## $H_a$ : Tax Rate is independent of  the log error of a property 

---
# Hypothesis 3
Log error is different among the counties of LA County, Orange County and Ventura County?

## $H_0$:  There is no significant difference in logerror for properties in LA County vs Orange County vs Ventura County

## $H_a$ :  Log error is significantly different among the counties of LA County, Orange County and Ventura County.
---
# Hypothesis 4
Does log error vary by the age of the house?

## $H_0$: Tax Value has a dependent relationship with a homes age.

## $H_a$ : Tax Value is independent of a homes age.
---
### Target variable
Log Error

### Need to haves (Deliverables):
acquire.py
prepare.py
Final_notebook.ipybn
this readme.md


### Nice to haves (With more time):




***

## <a name="findings"></a>Key Findings:
## There is more than one way to predict but simple is better and diving to deep will cause you to drown.

### Sqft plays a factor in value
### Number of Bathroom plays a factor in Value 
### Number of Bedrooms plays a factor in Value


[[Back to top](#top)]


***

## <a name="dictionary"></a>Data Dictionary  


### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|area|Area of the property in square feet|float64|
|assessmentyear|Year the taxes were assessed on the property|float64|
|bathroomct|Count of Bathrooms for the property|float64|
|bedroomcnt|Count of Bedrooms for the property|float64|
|county|County the property is in |object|
|fips|Federal Information Processing Standard code|int64|
|latitude|Latitude of the middle of the parcel divided by 10e6|float64|
|logerror|The log difference between Zillow's Zestimate and the property sale price|float64|
|longitude|Longitude of the middle of the parcel divided by 10e6|float64|
|yearbuilt|The Year the Property was Built|float64|
| ----- | ----- | ----- |
***
[[Back to top](#top)]
## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()


### Wrangle steps: 
Try to Make pretty pictures
Repeat until you get something you understand.

*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:

    - explore.py



### Takeaways from exploration:


***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]


# Stats Test 1

# Hypothesis 1
Is there a correlation between square footage of a home and log error?


## $H_0$: Square footage has a dependent relationship with the log error of a property 

## $H_a$ : Square footage  is independent of the log error of a property 
 

#### Alpha value:

- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
We reject the null hypothesis.

#### Summary:

***
----------
# Stats Test 2: 

# Hypothesis 2
Is there a relationship between tax value and log error?

## $H_0$: Tax Rate has a dependent relationship with the log error of a property 

## $H_a$ : Tax Rate is independent of  the log error of a property 


#### Results:
We reject the null hypothesis.


#### Summary:
While I still do not fully grasp this process it was completed
***
----------
# Stats Test 3: 

# Hypothesis 3
Log error is different among the counties of LA County, Orange County and Ventura County?

## $H_0$:  There is no significant difference in logerror for properties in LA County vs Orange County vs Ventura County

## $H_a$ :  Log error is significantly different among the counties of LA County, Orange County and Ventura County.
---

#### Alpha value:
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
We reject the null hypothesis.


#### Summary:
While I still do not fully grasp this process it was completed

***
----------
# Stats Test 4: 



# Hypothesis 4
Does log error vary by the age of the house?

## $H_0$: Tax Value has a dependent relationship with a homes age.

## $H_a$ : Tax Value is independent of a homes age.

#### Alpha value:
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
We reject the null hypothesis.


#### Summary:
While I still do not fully grasp this process it was completed
***
----------
## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Model Preparation:

### Baseline
    
- Baseline Results: 
Our baseline accuracy in all cases on the Dataset is :

RMSE Mean:
248150.10218076012
----------------
RMSE Median:
250857.8604903843

- Selected features to input into models:
    - features = Area, Bathrooms, Bedrooms

***

### Models and R<sup>2</sup> Values:
- Will run the following regression models:

    

- Other indicators of model performance with breif defiition and why it's important:

    


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

RMSE for Lasso & Lars
Training/In-Sample:  219234.24 
Validation/Out-of-Sample:  217879.84
R2: 0.22
_______________________________________________
RMSE for OLS using LinearRegression
Training/In-Sample:  219233.87 
Validation/Out-of-Sample:  217881.69
R2: 0.22
_______________________________________________
RMSE for Polynomial Model, degrees=2
Training/In-Sample:  219176.92 
Validation/Out-of-Sample:  217894.86
R2: 0.22





***

## <a name="conclusion"></a>Conclusion:
Sqft plays a factor in value,
Number of Bathroom plays a factor in Value,
Number of Bedrooms plays a factor in Value,
---
### Steps to Reproduce
Your readme should include useful and adequate instructions for reproducing your analysis and final report.

For example:

1)You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the titanic_db.passengers table. Store that env file locally in the repository.

2)clone my repo (including the wrangle.py, explore.py, and model.py) (confirm .gitignore is hiding your env.py file)

3)libraries used are pandas, matplotlib, seaborn, numpy, sklearn,scipy, math.

4)you should be able to run churn_report.


[[Back to top](#top)]
