### 1. Business Understanding 
#### In this application, we explore a dataset from kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure speed of processing. Our goal is to understand what factors make a car more or less expensive. As a result of our analysis, we will provide recommendations to the client, a used car dealership, as to what consumers value in a used car.

### 2. Data Preparation and Exploratory Analysis  
The frame comprises of car of 18 attributes as follows: 'id', 'region', 'price', 'year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'VIN', 'drive', 'size', 'type', 'paint_color', and 'state'. The cylinder attributed was converted from object to numeric data type. ID was dropped as VIN, a unique vehicle identification number was retained. The data was prepared by eliminating any duplicate rows with same VIN. A large number of categorical information around the model was available but was decided to be dropped as we focused on the analysis based on manufacturer.  


### 2.1 Exploratory Data Analysis
A prelimnary volume and price of the car datasheet suggested that 10.51 % of the population of car has price <$100, an impractical situation so we dropped it from the data analysis. On the higher side, 0.14% of the used car had price >$100k. 

![Number of Cars with Price   $100k by Manufacturer and Decade](https://github.com/amsc1985/CarPricePrediction/assets/37163650/2d643168-7984-4d77-a090-ffb7ff9d0a05)

These 160 used cars with `$>100k` price comprise of vintage cars from 1950 decade to the more luxury brand s like Porsche, Aston Martin, etc. Tracability to model the is not evaluated. Furthermore, of the 102022 used cars are available for sale with  1028  cars were made from 1905 to 2000. Ignoring the collectors market so we are dropping all cars that are less than 2000 make. Subesquent pricing analysis suggests, two price segment in the used car market. One segment concentrated around `$25,000` and another around `$44,000`. 

![Bimodal Distribution of Used Cars for Sale with two market around $25k   $44k](https://github.com/amsc1985/CarPricePrediction/assets/37163650/cca33102-fc5d-4016-8478-8b680fcfc403)

Market Preferences: The presence of two distinct peaks indicates a potential preference for cars in these two price ranges. This could be due to several factors:
Budget Constraints: Buyers might be concentrated in two budget categories, with a significant number looking for affordable cars around $25k and another group willing to spend more for potentially newer or feature-rich cars around $44k.
Car Lifespan: The two peaks might correspond to different lifespans for cars. Cars in the  `$25𝑘`  range might be older but still functional, while those around $44k could be newer models with more features. Demand and Availability: The relative heights of the peaks could indicate the relative availability of cars in each price segment. A higher peak could suggest more available cars in that price range.
Further Inferences (depending on data availability):

![Average Price by Car Age](https://github.com/amsc1985/CarPricePrediction/assets/37163650/cd5d46ba-778f-41be-88fb-933f0b6f534e)

Car Age: If you have data on car age alongside the price, you could analyze if the $25k segment tends to have older cars and the $44k segment tends to have newer cars, supporting the lifespan theory. Car Features: If information on car features is available, you could see if there are distinct feature sets associated with each price range. For instance, cars around $44k might have more luxury features or advanced technology compared to those around $25k. Overall, the bimodal distribution highlights the existence of potential price segments in the used car market, possibly reflecting buyer preferences, budget constraints, or car lifespans. By analyzing additional data about the cars, you can gain more insights into the reasons behind this distribution.

The price has positive corelation with the cylinders and negative with car age and odometer. 
![Correlation plot for Numerical Values of Used Car Data](https://github.com/amsc1985/CarPricePrediction/assets/37163650/46f46e94-80cb-48aa-878f-dc2b6d61d792)

Impact of Cylinders: In the join plot, the color gradient from blue (presumably lower number of cylinders) to red (presumably higher number of cylinders) suggests a possible trend. Cars with more cylinders (potentially larger or more powerful engines) might tend to be priced higher on average, even at similar mileage. However, due to the overlap in data points across colors, it's difficult to say definitively without more data or statistical analysis.
Other Observations: There is a cluster of data points, representing low-priced cars with very high mileage. These could be very old cars or ones that have been driven extensively. The spread of data points increases at higher mileage, indicating more variation in price for cars with very high mileage.

![Price vs  Odometer Reading (Color by Cylinders)](https://github.com/amsc1985/CarPricePrediction/assets/37163650/7009c1fa-b8bd-4181-b55e-1017f9e73501)

Impact of Drive Type: The color gradient from blue (presumably front-wheel drive) to red (presumably all-wheel drive) suggests a possible trend. Cars with all-wheel drive tend to be priced higher on average, even at similar mileage, compared to front-wheel drive cars. Rear-wheel drive cars seem to fall somewhere in between.
Other factors like car model, brand, condition, and features can also influence price.

![Price vs  Odometer Reading (Color by Drive Type)](https://github.com/amsc1985/CarPricePrediction/assets/37163650/39b2ed85-cfa1-4a57-8d0b-b0dea061ef52)


Price Distribution: The price distribution appears to be skewed towards lower prices. There might be a larger number of used cars concentrated in the lower price range.

Important Considerations: The strength of the correlation between price and mileage cannot be definitively determined from a scatter plot alone. Other factors like car model, brand, condition, and features can also influence price. 

Further Analysis: To get a more precise understanding of the relationship between price, mileage, and number of cylinders, we will perform statistical tests like linear regression next section. This will quantify the correlation and determine if the effect of cylinder count on price is statistically significant.
We can also consider creating separate scatter plots for different car models or brands to see if the price-mileage relationship varies across categories.

An exploratory data analysis using one hot encoding on condition and transmission where as target encoding was performed on feature like drive, type and state. The corresponding correaltion matrix is as follows. 
![Correlation plot for Numerical   Categorical Values of Used Car Data](https://github.com/amsc1985/CarPricePrediction/assets/37163650/c6efb603-be0e-4c4e-a846-2ff17fa62f64)


### 3. Modeling

Here, we build a number of different regression models with the price as the target.  We will explore different parameters and cross-validate our findings. For all models 20% of the data was assigned for test size. The prediciton is made on the log(price)

###### 3.1 Model Using `SequentialFeatureSelector`
A polynomial features used forward feature selection to select three features (`n_features_to_select = 3`) using a `LinearRegression` estimator to perform the feature selection on the training data. Not much improvement in the MSE and R2 was observed when using more than three degree polynomials. cylinders,	year^3, and year^2 odometer were the three attributes that gave training and test R2 of 59.5 and 60% respectively and the MSE of 0.23 each.

######  3.2 Cross-Validation with `SequentialFeatureSelector`
MSE with Cross-Validation with SequentialFeatureSelector was 0.25 and R2 of 56.2% using a `LinearRegression` estimator.

###### 3.3 Ridge regression model and its alpha parameter.
Ridge with 3 deg polynomial gave Train and Test MSE of 0.229 and 0.226. The corresponding Train and Test R^2 was 60.2% each

###### 3.4 Using `GridSearchCV`
`GridSearchCV` is used to search over different hyperparameter values within the `Ridge` estimator. As optimal value of alpha was found to be 10. Giving Ridge with Grid CV Test and Train MSE of 0.23 each. The Ridge with Grid CV Train R^2 was 59.80% and Test R^2 was 59.30%

###### 3.5 LASSO and Sequential Feature Selection
Lasso Train and Test MSE were 0.57 amd 0.576 respectively. The Training and Test R2 was 59.3% each. 
Results of Lasso to select features are subsequently used in a LinearRegression estimator. A Pipeline after transforming the features and before building a regression model.  The Permutation Importance are as follows
year    : 0.668 +/- 0.006
cylinders: 0.414 +/- 0.006
odometer: 0.095 +/- 0.001


A more elaborate permutation importance is giving upon doing one hot encoding and doing gridsearch CV using a pipeline. The R2 decreases to 38% with the inclusion of more data with the top five categories as 

<img width="1076" alt="pipeline information" src="https://github.com/amsc1985/CarPricePrediction/assets/37163650/c546a27f-e85a-4848-9ebc-ebf2359c2a69">

The top categories of permutation importance are 
cat__condition_excellent: 0.150 +/- 0.040
cat__condition_like new: 0.115 +/- 0.049
cat__drive_4wd: 0.106 +/- 0.025
cat__drive_fwd: 0.089 +/- 0.032
cat__title_status_rebuilt: 0.003 +/- 0.001
cat__paint_color_brown: 0.001 +/- 0.000
cat__manufacturer_lexus: 0.000 +/- 0.000

![Permutation Importances with manufacturer information](https://github.com/amsc1985/CarPricePrediction/assets/37163650/a1caa01d-72b6-4b46-a3be-029ad86b1ab8)

When the above is repeated by ignorining the manufacturer information, then, the R2 increases to 40% 

![Bar Graph of Coefficient Strength with manufacturer information](https://github.com/amsc1985/CarPricePrediction/assets/37163650/a2ebdceb-bea4-4453-921b-00bf29e450b9)
![Permutation Importances without manufacturer information](https://github.com/amsc1985/CarPricePrediction/assets/37163650/af1140ac-1ebd-4619-be75-f231d38211ba)


Overall the best prediction of car price ranging between `$30,000 - $100,000 for cars made from 2000 year was ~60% with MSE of 0.22 log car price using Ridge regression model of alpha =10 and Sequential Feature Selection with polynomial degree 3.

