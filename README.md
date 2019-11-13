# zestimate prediction
## please refer to .py file
## py notebook contains large amount of optimzation output and might not be good for reading.

(dataset larger than allowed size on GitHub)

Zillow is an online information platform for real estate with millions of home listed. Besides including the basic information like the area of the house and number of rooms, Zillow also built its own model to predict the sale price for property called Zestimate. However, a home's Zestimate does not always match the actual sale price of the property. The goal of this competition is to predict the error between the Zestimate and actual sale price. 

In this project, I built a prototype model using CatBoost and then conducted feature analysis and feature engineering to capture useful insights about the dataset and create some new features that I believed to be informative. But, the reality does not always match our hypothesis. Thus, I used a method called target permutation and null feature importance to select the informative features among the original and created features. Then I used Bayesian Optimization technique to select the optimal hyperparameter values to build final model. The output of final model hit a l1 score of 0.07508 on Kaggle private leaderboard and ranked 155th place out of 3775 teams, which is top 4.1%.
