# DataScience-PCAwithLogisticRegression
### I applied the following steps to the Breast Cancer Wisonsin Data-Set, which I found on Kaggle. This dataset predicts whether the cancer is benign or malignant.
#### PART1 <br> A. Read the data-set as a data frame.<br> B. Check whether your data types are correct or not by using str (structure) function.<br> C. Search whether any null values exist in the data-set.<br> D. Look at the correlation between the columns using the corrplot function in corrplot library<br> E. Discuss about multicollinearity.<br>
#### PART2 <br> A. Split your data into train and test data. <br> B. Use logistic regression in order to build a logistic model using glm function. <br> C. Discuss about the summary of your model. <br> D. Use predict function and then get the confusion matrıx. <br> E. Discuss about the performance measures. <br>
#### PART3 <br> A. Apply PCA choose your PC’s (explain the reason).<br> B. Use logistic regression with your PC’s in order to build a logistic model using glm function.<br> C. Discuss about the summary of your model. <br> D. Use predict function and then get the confusion matrıx. <br> E. Discuss about the performance measures. <br>

### After completing these steps, I conducted a comparison between 2.E and 3.E, leading to the following conclusions:
#### 2.E: <br> Accuracy: 0.961 <br> Precision: 0.947 <br> Recall/Sensitivity: 0.972 <br> F1 Score: 0.960 <br>

#### 3.E: <br> Accuracy: 1.0 <br> Precision: 1.0 <br> Recall/Sensitivity: 1.0 <br> F1 Score: 1.0 <br>

###   In general, the results of 3.E show that the model predicts both positive and negative classes better. This means that the model is accurate, precise, sensitive, and has a good F1 score. However, we see that the performance measures of 2.E are a bit lower. 
