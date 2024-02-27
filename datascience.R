
# PCA with LOGISTIC REGRESSION

# PART1

## A. Read the data-set as a data frame.

data <- as.data.frame(data)
print(head(data))


## B. Check whether your data types are correct or not by using str (structure) function.

print(str(data))
head(data$diagnosis)
data$diagnosis[data$diagnosis == "M"] <- "1"
data$diagnosis[data$diagnosis == "B"] <- "0"
data$diagnosis <- as.numeric(data$diagnosis)


## C. Search whether any null values exist in the data-set.


print(sum(is.na(data)))
data <- data[, colSums(is.na(data)) != nrow(data)]
print(sum(is.na(data)))


## D. Look at the correlation between the columns using the corrplot function in corrplot library.


library(corrplot)
corr_mat <- cor(data[, -ncol(data)])
corrplot(corr_mat, method="circle")


## E. Discuss about multicollinearity.

High correlations observed on the matrix indicate a strong relationship between the independent variables. In particular, correlations of 0.7 and above may increase the risk of multicollinearity. (id-diagnosis-texture_mean-radius_mean etc.)

The matrix can help us see multicollinearity effects. Particularly notable are high correlations, indicating that one variable can be derived from or is a combination of the others. (For example; The circle at the intersection of radius_mean and area_worst is dark, indicating that multicorreality is high. The high multicorreality shows us that these two values can derive from each other.)

If there are diagonally intersecting lines within the circle, this may indicate that there is a linear relationship between the variables. This may indicate a problem of multicollinearity because one variable can be explained by the others.

The distance between variables within the circle indicates that the correlation between these variables decreases with distance. However, if there is still a dense circle, this may indicate that the risk of multicollinearity remains.

# PART2

## A. Split your data into train and test data.


library(caret)
set.seed(123)
trainIndex <- createDataPartition(data$diagnosis, p=0.7, list=FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]


## B. Use logistic regression in order to build a logistic model using glm function.


model <- glm(diagnosis ~ ., data=train_data, family=binomial)


## C. Discuss about the summary of your model.

summary(model)


The model displays z-values close to zero, tiny standard errors, and p-values close to 1 for several coefficients. This shows problems with the model or potential collinearity issues among predictor variables. We have to examine the data consider variable selection or transformation to address.

## D. Use predict function and then get the confusion matrıx.


predictions <- predict(model, newdata = test_data, type = "response")
threshold <- 0.5
predicted_classes <- ifelse(predictions > threshold, 1, 0)
conf_matrix <- table(Actual = test_data$diagnosis, Predicted = predicted_classes)
conf_matrix


## E. Discuss about the performance measures.

True Negatives (TN): 93

False Positives (FP): 4

False Negatives (FN): 2

True Positives (TP): 71

Accuracy -\> (TN + TP) / (TN + FP + FN + TP) = (93 + 71) / (93 + 4 + 2 + 71) ≈ 0.961

Precision -\> TP / (FP + TP) = 71 / (4 + 71) ≈ 0.947

Recall/Sensitivity -\> TP / (FN + TP) = 71 / (2 + 71) ≈ 0.972

F1 Score -\> 2 \* (Precision \* Recall) / (Precision + Recall) = 2 \* (0.947 \* 0.972) / (0.947 + 0.972) ≈ 0.960

According to the results of the above metrics, the performance measures we obtained are generally high and evenly distributed.

# PART3

## A. Apply PCA choose your PC's (explain the reason)


numeric_data <- data[, sapply(data, is.numeric)]
scaled_data <- scale(numeric_data)
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)
summary(pca_result)
num_pcs <- 5
selected_pcs <- pca_result$x[, 1:num_pcs]


Choosing a specific number of principal components helps us build our model while preserving the most important features that represent the data. In this way, we obtain a representation of the model with fewer dimensions but containing important information. The number of principal components chosen can affect the complexity of the resulting representation and the performance of the model, so we chose a small number.

### B. Use logistic regression with your PC's in order to build a logistic model using glm function


numeric_data <- data[, sapply(data, is.numeric)]
scaled_data <- scale(numeric_data)
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)
num_pcs <- 5
selected_pcs <- pca_result$x[, 1:num_pcs]
pca_data <- data.frame(selected_pcs, diagnosis = as.factor(data$diagnosis))
set.seed(123)
train_indices <- sample(1:nrow(pca_data), 0.7 * nrow(pca_data))
train_data <- pca_data[train_indices, ]
test_data <- pca_data[-train_indices, ]
pca_model <- glm(diagnosis ~ ., data = train_data, family = binomial)


## C. Discuss about the summary of your model.

summary(pca_model)


## D. Use predict function and then get the confusion matrıx.

predictions <- predict(pca_model, newdata = test_data, type = "response")
threshold <- 0.5
predicted_classes <- ifelse(predictions > threshold, 1, 0)
conf_matrix <- table(Actual = test_data$diagnosis, Predicted = predicted_classes)
conf_matrix


## E. Discuss about the performance measures.

True Negatives (TN): 73

False Positives (FP): 0

False Negatives (FN): 0

True Positives (TP): 98

Accuracy -\> (TN + TP) / (TN + FP + FN + TP) = 1.0

Precision -\> TP / (FP + TP) = 1.0

Recall/Sensitivity -\> TP / (FN + TP) = 1.0

F1 Score -\> 2 \* (Precision \* Recall) / (Precision + Recall) = 1.0

Since all of the above metrics received the value of 1.0, we can say that the model performs very well.

## F. Compare results 2.E and 3.E

2.E:
  
  Accuracy: 0.961

Precision: 0.947

Recall/Sensitivity: 0.972 F1 Score: 0.960

3.E:
  
  Accuracy: 1.0

Precision: 1.0

Recall/Sensitivity: 1.0

F1 Score: 1.0

In general, the results of 3.E show that the model predicts both positive and negative classes better. This means that the model is accurate, precise, sensitive, and has a good F1 score. However, we see that the performance measures of 2.E are a bit lower.
