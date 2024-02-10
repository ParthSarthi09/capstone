# capstone
HOTEL BOOKING CANCELLATION PREDICTION

 





HOTEL BOOKING CANCELLATION PREDICTION

Post Graduate Program in Data Science Engineering



Location: Bangalore Batch : PGP DSE- APR 22

Submitted by
EVELINE JEMIMA SONA PAUL
HARSH PRIYADHARSHAN
ANKIT KUMAR
PARTHO SARATHI BISWAS

Mentored by
PRATIK SONAR 

Table of Contents

Sl. NO	Topic	Page No
1	INTRODUCTION	1
2	BUSINESS PROBLEM STATEMENT	2
3	LITERATURE REVIEW	3
4	DATASET INFORMATION	4
5	VARIABLE IDENTIFICATION	5
6	APPROACH	6
7	DATA PREPROCESSING	7
8	STATISTICAL ANALYSIS	11
9	UNIVARIATE, BIVARIATE & MULTIVARIATE ANALYSIS	13
10	ENCODING	22
11	SCALING &
TRANSFORMATION	23
12	CHECKING FOR CLASS IMBALANCE	24
13	MODEL BUILDING	25
14	CONCLUSION	43
15	REFERENCES	44
 

INTRODUCTION:

Hotel booking cancellation prediction is a critical facet of the hospitality industry, influencing operational efficiency, revenue management, and customer satisfaction. This project aims to employ advanced machine learning techniques to forecast the likelihood of hotel booking cancellations, empowering establishments to make informed decisions and optimize resource utilization.
The significance of this project lies in its potential to revolutionize the way hotels manage their reservations. By leveraging historical booking data, alongside a multitude of variables such as booking lead time, seasonality, room type, and customer demographics, predictive models can be developed to anticipate the probability of a reservation being canceled. This foresight equips hoteliers with valuable insights to adapt their strategies, allocate resources effectively, and potentially mitigate revenue loss.
The methodology involves the utilization of various machine learning algorithms, such as logistic regression, decision trees, random forests to analyze and predict cancellation . Data preprocessing and feature engineering play a crucial role in enhancing the accuracy of these models. By identifying patterns and correlations within the data, the models can effectively learn and predict the likelihood of a booking getting canceled.
Moreover, this project prioritizes the interpretability of the models to enable hotel management to comprehend the reasons behind these forecasts. By understanding the key factors contributing to booking cancellations, hotels can proactively implement strategies to reduce cancellation rates. Whether it's through personalized offers, flexible cancellation policies, or targeted customer engagement, this knowledge can drive proactive measures to decrease cancellations and enhance customer satisfaction.
The implications of accurate cancellation prediction are far-reaching. It allows hotels to optimize their overbooking strategies, staff allocation, and inventory management, thereby reducing revenue loss resulting from cancellations. Additionally, it enables personalized customer service, as hotels can anticipate and cater to the needs of guests who are more likely to cancel, potentially retaining their business through tailored approaches.
Ultimately, the project endeavors to create a robust, adaptable, and scalable solution that can be integrated into hotel management systems. By harnessing the power of predictive analytics, this project aims to assist hotels in making data-driven decisions, enhancing operational efficiency, and ultimately improving the overall guest experience.

BUSINESS PROBLEM STATEMENT:

Business Problem Understanding: The business problem lies in the inability of hotels to accurately predict booking cancellations, leading to revenue loss and operational inefficiencies. The lack of a reliable forecasting mechanism hinders effective resource allocation, overbooking strategies, and personalized customer service. This project aims to address this challenge by leveraging machine learning to develop a predictive model. The goal is to empower hotels with actionable insights, enabling them to proactively manage cancellations, optimize revenue streams, and enhance overall operational efficiency. The solution holds the potential to transform the hospitality industry by providing a data-driven approach to mitigate the impact of unpredictable booking cancellations
Business Objective: The primary business objective for implementing a hotel booking cancellation prediction system is to enhance operational efficiency, maximize revenue, and improve overall customer satisfaction within the hospitality industry. The specific business objectives include:
1.	Optimizing Resource Allocation:
•	Improve efficiency in staff management, housekeeping, and other operational aspects by accurately predicting booking cancellations.
•	Minimize the impact of unexpected cancellations on staffing levels and resource allocation, ensuring optimal utilization of hotel resources.
2.	Revenue Maximization:
•	Implement proactive overbooking strategies based on accurate cancellation predictions to capitalize on potential demand without risking overcapacity.
•	Identify opportunities to offer additional services or promotions to guests at risk of canceling, thereby retaining revenue that might otherwise be lost.
3.	Data-Driven Decision Making:
•	Foster a culture of data-driven decision-making within the organization by leveraging predictive analytics to anticipate booking cancellations.
•	Provide actionable insights to hotel management for strategic planning and the development of targeted marketing and retention initiatives.
4.	Enhancing Customer Experience:
•	Implement personalized services and retention strategies for guests identified as likely to cancel, improving overall customer satisfaction.
•	Proactively communicate with guests facing potential cancellations, offering flexible options and demonstrating a commitment to customer-centric practices.

DATASET INFORMATION
The dataset has been published in Kaggle. The dataset contains 32 attributes with 119390 observations. The datasets contain a mix of features with floating-point, integer, and string values. For this study, we intend to use the most common data attributes that can be readily available for any business. Hence, we have chosen 'reserved_room_type', 'assigned_room_type', 'booking_changes' as our predicting variables and ‘is_cancelled’ as our response variable. Our target variable is labeled with two classes. Hence, this scenario falls under the binary classification problem.

The booking changes feature indicates : Number of changes/amendments made to the booking from the moment the booking was entered on the Property Management System until the moment of check-in or cancellation. Calculated by adding the number of unique iterations that change some of the booking attributes, namely: persons, arrival date, nights, reserved room type or meal.The ‘lead time’ feature : Number of changes/amendments made to the booking from the moment the booking was entered on the Property Management System until the moment of check-in or cancellation. Calculated by adding the number of unique iterations that change some of the booking attributes, namely: persons, arrival date, nights, reserved room type or meal.
PreviousCancellations feature indicates INumber of previous bookings that were canceled by the customer prior to the current booking. In case there was no customer profile associated with the booking, the value is set to 0. Otherwise, the value is the number of bookings with the same customer profile created before the current booking and canceled. 

VARIABLE IDENTIFICATION
 Variables: The 32 columns are listed below :
 
Fig: Variables
 

APPROACH
The approach to build a Hotel Booking Cancellation Prediction model typically involves the following steps:

1.	Data collection: , the project team will collect the relevant data from various sources. They will ensure the data is accurate, complete, and relevant to the project goals.

2.	Data preprocessing: Clean and preprocess the data to remove outliers, missing values, and other errors. Aggregate the data into a format suitable for training machine learning models.

3.	Feature engineering: Create new features or transform existing ones that can improve the accuracy of the model. For example, features such as countries, arrival_date_month, market_segment may be included to help classify hotel booking cancellation.

4.	Model selection: Select an appropriate machine learning algorithm for the Hotel Booking Cancellation Prediction task. Commonly used algorithms include logistic regression, decision trees, and random forests.

5.	Model training: Train the selected machine learning model on the preprocessed data, using techniques such as cross-validation to optimize model parameters.

6.	Model evaluation: Evaluate the trained model's accuracy and performance using metrics such as accuracy,precision and recall.

7.	Model deployment: Deploy the trained model in a production environment to make predictions on new data. Integrate the model with inventory management systems to provide real-time insights and recommendations.

8.	Model monitoring and updating: Monitor the model's performance over time and retrain or update the model as needed to ensure its accuracy and effectiveness.

Overall, the approach to Hotel Booking Cancellation Prediction involves a combination of data processing, feature engineering, machine learning, and integration with inventory management systems to provide real-time insights and recommendations to businesses.
 

TARGET VARIABLE

The target variable of the above dataset is is_cancelled. We have to predict whether the booking will get cancelled or not. In the above dataset, 63% of bookings are not going to get cancelled and 37% of are going for cancellation. We observe that this data is  very much balanced.

DATA PREPROCESSING:

Data pre-processing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model. When creating a machine learning project, it is not always a case that we come across the clean and formatted data. And while doing any operation with data, it is mandatory to clean it and put in a formatted way. So for this, we use data pre-processing task.

A real-world data generally contains noises, missing values, and maybe in an unusable format which cannot be directly used for machine learning models. Data pre- processing is required tasks for cleaning the data and making it suitable for a machine learning model which also increases the accuracy and efficiency of a machine learning model.
 

Missing Value Treatment:
 
The next step of data pre-processing is to handle missing data in the datasets. If our dataset contains some missing data, then it may create a huge problem for our machine learning model. Hence it is necessary to handle missing values present in the dataset.
Fig : Missing Value

In dataset we have missing values in some features. children feature is integer datatype, replacing  nan values with 0. Filling it with median imputation. Country column is categorical. So we are filling the nan values with mode value. We are dropping the column company since it is have null values of 94%.
 

 Imputation:
Fig: Heatmap of variables to show the null values

Imputing missing values is an important step in data preprocessing, and the choice of imputation method depends on the characteristics of the data and the nature of the missingness. Median imputation is one of the simplest and most commonly used methods for imputing missing values, particularly when the data contains outliers. The median is a robust measure of central tendency that is not affected by extreme values or outliers in the data. Therefore, median imputation can be a useful method when outliers are present in the data, as it provides a reasonable estimate of the missing value without being unduly influenced by extreme values.



However, it is important to note that median imputation has some limitations. For example, it assumes that the data are approximately normally distributed, which may not always be the case. Additionally, median imputation can introduce bias in the estimates of other parameters, such as variances and covariances.Therefore, while median imputation can be a useful method for imputing missing values in the presence of outliers, it is important to carefully consider the assumptions and limitations of the method and to explore alternative methods if necessary.
 

Checking for outliers:
 
Fig: Boxplot of numerical variables

The box plots above are plotted for some numerical columns. 

 

Nearly half of our dataset is having outliers. So we have taken Q1 as 0.05 and Q3 as 0.95 and treated the outliers.





DROPPING OF COLUMNS
 

We dropped some of the columns whose standard deviation value is 0.

STATISTICAL ANALYSIS

Chi-square test of independence:

Null hypothesis: There is no relationship between the two categorical variables. Alternative hypothesis: There is a relationship between the two categorical variables.
Here , we passed all the categorical variables with target variable inside for loop. To check whether there is relationship between all categorical variables and target variable.
Here, we found out that all the categorical variables are significant.


 
Fig: Chi2 contingency Test

Mann-Whitney Test

To check the significance of all numeric variables with target variable, we use the parametric test ttest_ind or Wilcoxon test.The Wilcoxon signed-rank test, sometimes simply referred to as the Wilcoxon test, is a non-parametric statistical test used to compare two related or matched samples. It's an alternative to the t-test when the assumptions of normality are not met. Here we conclude that all the numeric variables are significant.
 

Fig: Mann-Whitney test
 

UNIVARIATE,BIVARIATE&MULTIVARIATE ANALYSIS
Univariate Analysis
Summary Statistics of Numerical Variable:


 
Fig:5 point summary


There is a large difference between mean and median values , indicating that the variables are highly skewed. There is a large difference between the maximum and mean values, it typically indicates that there are a few extremely large values in the dataset that are significantly driving up the maximum value.
 

KDE PLOT
 
Fig: KDE Plot of numerical variables


The arrival_date_day_of_month is moderately skewed. The variable adults  and arrival_date_year are negatively skewed. Rest of the variables are highly positively skewed.





















Summary Statistics of Categorical Variable:
 
Fig: Summary Statistics of Categorical variables


There are eleven categorical variables: " hotel ", " arrival_date_month", " meal", " market_segment", "distribution_channel", " reserved_room_type",” deposit_type”,” customer_type”,” reservation_status”,” continents” and "assigned_room_type. We can infer that City Hotel count is above 60000.Online Travel Agencies is the most preferred.Travel Agents and Tour Operators are widely utilized distribution channels. Most of the hotels are in Europe.Transient customer type is more. In the month of August, more customers came.The room type 'A' is assigned to more customers.Bed and Breakfast is taken by majority of the people. 

BIVARIATE ANALYSIS
 


Based on the countplot of reservation_status with is_canceled, we can drop the column reservation_status because it resembles the target column which we can see in the plot. 

BOX PLOT
 
Fig: Boxplot of quantitative-target (is_canceled)

The box plots is plotted between all numeric variables and target variable ‘is_cancelled’. If the median of the categories of categorical variable is same, then we can conclude that the categorical variable is insignificant. 


Heat Map
 
Fig: Heatmap (Numerical vs Numerical)


 

Fig: Heatmap(Correlation >0.65)
 

A heat map is a type of data visualization that uses color-coded squares or rectangles to represent the values of a two-dimensional dataset. It is particularly useful for visualizing large datasets and for identifying patterns or trends in the data.In a heat map, each row and column of the dataset is represented by a separate square or rectangle, and the color of the square represents the value of the corresponding cell in the dataset. The color scale typically ranges from a low value (e.g., blue or green) to a high value (e.g., red or yellow), with intermediate values represented by shades of the intermediate colors. They can also be used to visualize patterns or relationships in the data, such as correlations between variables or clusters of similar observations.

From the correlation matrix , we can easily interpret that none of the features are correlated with each other and there’s absence of multicollinearity.
ENCODING:
Encoding refers to the process of converting data from one representation to another. In the context of machine learning, encoding is often used to convert categorical variables (e.g., colors, categories, labels) into a numerical representation that can be used by algorithms. This is important because many machine learning algorithms require numerical inputs.

There are several methods of encoding categorical variables, including one-hot encoding, label encoding, and ordinal encoding. One-hot encoding involves creating a binary column for each category, where the column is 1 if the category is present and 0 otherwise. Label encoding involves assigning each category a numerical label (e.g., 0, 1, 2, etc.), which can be problematic if the labels have an order or hierarchy to them. Ordinal encoding is similar to label encoding, but it preserves the order of the categories by assigning each category a numerical value based on its position in a predetermined order.
 

SCALING & TRANSFORMATION
Scaling and transformation are two common data preprocessing techniques used to prepare data for analysis or modelling . Scaling involves transforming the values of the features in a dataset to a common scale, typically by subtracting the mean and dividing by the standard deviation. This is done to ensure that all features have the same scale and range, which can be important for certain algorithms, such as those that are distance-based. Transformation involves applying a mathematical function to the values of the features in a dataset to transform them in a specific way.

Robust scaling is a data normalization technique used in machine learning to scale numerical data so that it has a mean of 0 and a standard deviation of 1, while being robust to the presence of outliers. It is similar to standard scaling, but instead of using the mean and standard deviation of the data, it uses the median and interquartile range (IQR).

Yeo-Johnson Power Transformation is a data transformation technique used to normalize numerical data that may not follow a normal distribution. It is an extension of the Box-Cox transformation that can handle both positive and negative values. The Yeo-Johnson transformation applies a power transformation to the data using a lambda parameter that is chosen to maximize the normality of the transformed data.

 

We have done robust scaler here. 

CHECKING FOR CLASS IMBALANCE
In this case, the '0' class represents about 63% of the samples, while the '1' class represents only 37% of the samples. There is no class imbalance present here.
MODEL BUILDING
Step by step approach for model building: -

1.	Data Preprocessing: Clean and preprocess the data by handling missing values, encoding categorical features, scaling numerical features, and removing outliers.
2.	Data Splitting: Split the data into training, validation, and testing sets. The training set is used to train the model, the validation set is used to tune the hyperparameters of the model, and the testing set is used to evaluate the final performance of the model.
3.	Feature Engineering: Create new features from existing ones that can improve the model's predictive power. This can involve transforming or combining features, extracting new features from text or images, or engineering domain-specific features.
4.	Model Selection: Select an appropriate machine learning model for the problem at hand, based on the type of data, the size of the dataset, and the performance metrics. Consider using simple models first, and gradually increasing complexity if necessary.
5.	Hyperparameter Tuning: Tune the hyperparameters of the chosen model to optimize its performance on the validation set. This can involve selecting the best learning rate, regularization parameter, activation function, number of hidden layers, and other parameters.
6.	Model Training: Train the final model on the entire training set using the optimized hyperparameters.
7.	Model Evaluation: Evaluate the final model on the testing set using appropriate performance metrics such as accuracy, precision, recall, F1-score, or AUC-ROC curve. Compare the performance of the model with the baseline and previous state-of-the-art methods.
8.	Model Deployment: Deploy the final model in a production environment, such as a web application, mobile app, or API, and monitor its performance over time. Update the model as necessary to improve its accuracy or adapt to changing data distributions.

Overall, this step-by-step approach provides a structured and systematic way to build, evaluate, and deploy machine learning models, and can help ensure the quality and robustness of the models.
 

BASE MODEL (Logistic Regression Model)
LOGISTIC REGRESSION SUMMARY:



Based on the p-values, several of the predictor variables appear to be significantly associated with the probability of an item going on backorder, including "national_inv", "lead_time", "in_transit_qty", "forecast_3_month", "forecast_9_month", "sales_1_month", "sales_3_month", "sales_9_month", "min_bank", "pieces_past_due", "perf_6_month_avg", "local_bo_qty", "deck_risk", "ppap_risk", and "stop_auto_buy".The coefficients for "potential_issue", "oe_constraint", and "rev_stop" do not appear to be significant as their p-values are greater than 0.05. However, it's important to note that the model failed to converge and further analysis may be needed to determine the robustness of the results.
 

TEST REPORT FOR BASE MODEL:

Fig: Test Report (Base Model)

The precision of the model for class 0 (negative instances) is 0.91, which means that out of all instances predicted as negative, 91% were actually negative. The recall for class 0 is 0.96, which means that out of all actual negative instances, the model correctly identified 96%. The F1-score for class 0 is 0.93, which is the harmonic mean of precision and recall for class 0.


Similarly, for class 1 (positive instances), the precision of the model is 0.71, which means that out of all instances predicted as positive, 71% were actually positive. The recall for class 1 is 0.49, which means that out of all actual positive instances, the model correctly identified 49%. The F1-score for class 1 is 0.58.


The overall accuracy of the model is 0.88, which means that the model correctly predicted 88% of the instances in the test set.


The macro average of precision, recall and F1-score for both classes is 0.81, 0.73 and 0.76 respectively. The weighted average of precision, recall and F1-score is 0.87, 0.88 and 0.87 respectively, taking into account the class imbalance present in the data.


Based on the test report, the model appears to perform better on negative instances than positive instances. There may be further room for improvement in the model's performance, especially for positive instances.
 

CONFUSION MATRIX


Fig: Confusion Matrix


In this case, your logistic regression model produced the following results:


There were 28,466 true negatives (i.e., the model correctly predicted that 28,466 instances were negative). There were 1,177 false positives (i.e., the model incorrectly predicted that 1,177 instances were positive). There were 2,969 false negatives (i.e., the model incorrectly predicted that 2,969 instances were negative). There were 2,869 true positives (i.e., the model correctly predicted that 2,869 instances were positive). The confusion matrix can be used to calculate various performance metrics for your logistic regression model, such as accuracy, precision, recall, and F1 score.
 

PERFORMANCE METRICS:

Fig: Performance Metrics(Base Model)

The model's accuracy is 0.883, which indicates that the model correctly classified 88.3% of the instances in the dataset. The precision of the model is 0.709, which means that out of all instances predicted as positive, 70.9% were actually positive. The recall of the model is 0.491, which means that out of all actual positive instances, the model correctly identified 49.1%. The F1-score of the model is 0.581, which is the harmonic mean of precision and recall, and provides a balance between these two metrics. The AUC-ROC score of 0.726 indicates that the model has a moderate ability to distinguish between positive and negative instances.

The kappa value of 0.515 suggests that there is moderate agreement between the predictions made by the model and the actual values. This metric is useful in assessing inter-rater agreement, and values close to 1 indicate strong agreement, while values close to 0 indicate chance agreement. AUC-ROC is a metric that measures the area under the receiver operating characteristic (ROC) curve. The ROC curve is a plot of true positive rate (recall) versus false positive rate (1 - specificity) at various thresholds. In this case, the AUC-ROC is 0.725, indicating that the model performs moderately well at distinguishing between positive and negative instances.

The computed cross-entropy loss on the test data is 0.2624, which is a measure of how well the logistic regression model fits the data and predicts the probability of the target variable (went_on_backorder). The lower the cross-entropy loss, the better the model. In this case, the obtained value of 0.2624 is relatively low, indicating that the logistic regression model has a good fit to the data and can predict the probability of a product going on backorder with reasonable accuracy.

Overall, the model appears to have moderate performance in terms of accuracy, precision, recall, F1- score, AUC-ROC, kappa value and cross-entropy. There may be room for improvement in the model's performance, especially in identifying positive instances.
 

ROC CURVE
Fig: ROC CURVE

The AUC of your logistic regression model on the test data is approximately 0.73. AUC-ROC is a metric that measures the area under the receiver operating characteristic (ROC) curve. The ROC curve is a plot of true positive rate (recall) versus false positive rate (1 - specificity) at various thresholds. In this case, the AUC-ROC is 0.7258, indicating that the model performs moderately well at distinguishing between positive and negative instances.
 

Cross-Validation

Fig: Cross-Validation


These scores are relatively consistent, with a mean score of approximately 0.881 and a standard deviation of 0.0015. This suggests that the model's performance is consistent across different subsets of the training data, and there is no significant overfitting or underfitting. However, the performance of the model on the test dataset should also be evaluated to assess its generalization ability.
Kappa Score

Fig: Kappa Score
The kappa statistic is a measure of agreement between the predicted and actual classifications, which takes into account the possibility of agreement occurring by chance. The kappa value ranges from -1 to 1, where a value of 1 indicates perfect agreement, 0 indicates agreement by chance, and -1 indicates complete disagreement.
In this case, the kappa value is 0.5152, which indicates moderate agreement between the predicted and actual classifications. This means that the model's predictions are better than chance, but there is still some room for improvement in its performance.
 

Random Forest Classifier:


Fig: Test Report (Random Forest Classifier)
The RFC model appears to be a very effective approach for predicting the target variable. The model has performed very well on both the training and test sets. The precision, recall, and F1-score are all very high, indicating that the model is accurately identifying both positive and negative examples. The accuracy on the training set is 100%, which is a good indication that the model has learned the training data very well.

The accuracy on the test set is slightly lower at 98%, but this is still a very good result. The precision for the negative class (0) is very high at 0.99, which means that when the model predicts an example as negative, it is almost always correct. The precision for the positive class (1) is slightly lower at 0.96, which means that when the model predicts an example as positive, there is a higher chance of it being incorrect. The recall for the positive class is also lower at 0.93, which means that the model is missing some positive examples.

Overall, these results indicate that the model is performing very well and is accurately identifying both positive and negative examples. However, there is some room for improvement in terms of the model's performance on the positive class, which could potentially be improved through further tuning or adjustments to the model architecture.
 

PERFORMANCE METRICS:
 
Fig: Performance Metrics (Random Forest)

The random forest model appears to be performing very well on the given data. The accuracy of the model is 0.9827, which means that the model is able to correctly classify a large proportion of the observations. The precision of the model is 0.9639, which indicates that when the model predicts a positive outcome, it is correct about 96.39% of the time. The recall of the model is 0.9296, which means that the model is able to correctly identify 92.96% of the positive cases. The F1-score of the model is 0.9465, which is a weighted average of the precision and recall, and provides an overall measure of the model's performance.

The AUC-ROC of the model is 0.9614, which indicates that the model is able to distinguish between the positive and negative cases with a high degree of accuracy. The kappa value of the model is 0.9361, which indicates a high level of agreement between the predicted and actual classifications. Finally, the cross- entropy of the model is 0.0768, which is a measure of the model's uncertainty about its predictions, and a lower value indicates better performance. Overall, the random forest model appears to be a very good fit for the given data.
 

Gradient Boosting Classifier:
Fig: Test Report(Gradient Boosting Classifier )

The training and testing reports for gradient boosting classifier (gbc) show that the model has an overall good performance, with an accuracy of 0.96 in both the training and testing sets. However, compared to the random forest classifier, the gbc model seems to have slightly worse performance in terms of precision, recall, and f1-score for the positive class (class 1), which is the class of interest since it represents the minority class. The gbc model also has a lower AUC-ROC and kappa value than the random forest classifier, indicating that the gbc model is less accurate in distinguishing between the two classes.

Overall, the gbc model may still be a good option for classification, but it may need further optimization or feature engineering to improve its performance.
 

PERFORMANCE METRICS:
 

Fig: Performance Metrics(Gradient Boosting Classifier)

Based on the metrics you provided, it seems that you trained a Gradient Boosting Classifier (GBC) model. The overall accuracy is 0.958, which is quite good. Precision, recall, and F1-score are also reasonably high, indicating that the model performs well in both identifying positive cases and avoiding false positives.


The AUC-ROC score of 0.900 indicates that the model is capable of distinguishing between positive and negative classes, and the kappa value of 0.842 suggests that there is substantial agreement between the predicted and true labels. Finally, the cross-entropy value of 0.127 suggests that the model has a low loss value and is well-calibrated. Overall, the metrics suggest that the GBC model is performing well on the task at hand.
 

Comparison of Performance of Different Models

Fig: Overall Comparison of Performance of all Models

Based on the evaluation metrics, the XG Boost model has the highest accuracy (0.986), recall (0.929), precision (0.982), F1 score (0.955), kappa score (0.946), and AUC ROC score (0.963) among all the models evaluated. This indicates that the XG Boost model is the best-performing model for the given task.


The Random Forest model comes in second place with an accuracy of 0.983, recall of 0.933, precision of 0.965, F1 score of 0.949, kappa score of 0.939, and AUC ROC score of 0.963. This model can also be considered a good option for the given task.


The Decision Tree model has an accuracy of 0.967, recall of 0.910, precision of 0.891, F1 score of 0.901, kappa score of 0.881, and AUC ROC score of 0.944. While this model performs lower than the top two models, it is still a reasonable option.


The Gradient Boosting model has an accuracy of 0.959, recall of 0.814, precision of 0.927, F1 score of 0.867, kappa score of 0.843, and AUC ROC score of 0.901. This model performed worse than the top three models, but still has a decent performance.
 

The KNN model has an accuracy of 0.951, recall of 0.961, precision of 0.786, F1 score of 0.865, kappa score of 0.835, and AUC ROC score of 0.955. This model has a high recall but a lower precision, making it more suitable for cases where identifying all true positives is more important than avoiding false positives.


The AdaBoost model has an accuracy of 0.934, recall of 0.718, precision of 0.858, F1 score of 0.782, kappa score of 0.743, and AUC ROC score of 0.847. This model has a lower performance than the top models but is still a reasonable option for the given task.


The SVM model has an accuracy of 0.896, recall of 0.571, precision of 0.740, F1 score of 0.645, kappa score of 0.585, and AUC ROC score of 0.766. This model has the lowest performance among all the models evaluated and may not be suitable for the given task.


The Logistic Regression model has an accuracy of 0.883, recall of 0.491, precision of 0.709, F1 score of 0.581, kappa score of 0.515, and AUC ROC score of 0.726. This model also has a low performance among the evaluated models.


The Naive Bayes model has the lowest performance among all models with an accuracy of 0.336, recall of 0.994, precision of 0.198, F1 score of 0.330, kappa score of 0.077, and AUC ROC score of 0.600. This model may not be suitable for the given task.


In summary, the XG Boost model is the best-performing model for the given task, followed by the Random Forest model, while the SVM and Naive Bayes models have the lowest performance among all the models evaluated. However, the choice of the best model depends on the specific problem and the data used, so it's important to thoroughly evaluate different models and choose the one that performs the best on the given task.
 

HYPERPARAMETER TUNING
1.	RandomForestClassifier Using Randomized Search CV:


Fig: Hyperparameter Tuning (RandomForestClassifier)
The best hyperparameters found for the Random Forest model are:
•	bootstrap: False
•	criterion: 'gini'
•	max_ depth: None
•	max_ features: 'log2'
•	min_ samples_ leaf: 1
•	min_ samples_ split: 2
•	n_ estimators: 225

These hyperparameters were found by tuning the model using a specific dataset and evaluation metric. The best score achieved by this model on the given dataset is 0.9836813932255957, which indicates a high level of performance. The evaluation metrics for this model are as follows:
Accuracy: 0.9848369549899947
Precision: 0.9690265486725663
Recall: 0.9378211716341213
F1-score: 0.9531685236768802
AUC-ROC: 0.9659587928136534
The high accuracy score indicates that the model is performing well in correctly classifying the target variable. The high precision score suggests that the model has a low false positive rate, meaning that when it predicts a positive outcome, it is usually correct. The high recall score indicates that the model has a low false negative rate, meaning that when a positive outcome is present, the model is good at identifying it. The F1-score, which is a harmonic mean of precision and recall, is also high, indicating an overall good performance of the model.
 

The AUC-ROC score of 0.9659587928136534 indicates that the model has a high ability to distinguish between positive and negative classes. Overall, these evaluation metrics suggest that the Random Forest model with the given hyperparameters is performing well and is a good fit for the given dataset.

2.	XG BOOST Using GridSearchCV


Fig: XG Boost using GridSearcb CV

Based on the information provided, it appears that a classification task was performed using the XG Boost algorithm, and the best hyperparameters for the model were determined to be a learning rate of 0.3, a maximum depth of 7, and 100 estimators. The model was trained on a dataset, and its performance was evaluated on a validation set using various metrics such as ROC AUC, accuracy, precision, recall, F1- score, Cohen's kappa, and log loss. The best values for these metrics were obtained on the validation set, and they suggest that the XG Boost model is performing well. The model was then tested on a separate test set, and its performance was evaluated using the same metrics. The test results indicate that the XG Boost model is generalizing well, as it achieved similar high performance on the test set.

Overall, based on the provided information, it appears that the XG Boost model is an effective machine learning algorithm for the given classification task, as it achieved high performance on both the validation and test sets. The best hyperparameters and performance metrics of the model can be used as a reference for future model training and evaluation.
 

VIF - Multicollinearity Between the Independent Variables

VIF (Variance Inflation Factor) is a measure of multicollinearity between independent variables in a model. It quantifies the extent to which the variance of the estimated coefficients is increased due to multicollinearity in the data. In the case of a binary classification problem, VIF can still be used to detect multicollinearity between independent variables. Multicollinearity can be a problem in any regression model, regardless of whether the dependent variable is continuous or binary.
To detect multicollinearity in a binary classification problem, one can calculate the VIF for each independent variable in the model. A VIF value greater than 5 or 10 is often considered to be indicative of high multicollinearity. If high multicollinearity is detected, it may be necessary to remove some of the independent variables from the model or to use a different modeling technique that is less sensitive to multicollinearity.
 	 

Fig: Before VIF	Fig: After VIF
 

Comparison of Performance of Different Models After VIF
Fig: Model Performance after VIF


From the given results, it can be observed that XG Boost and Random Forest models have the highest accuracy, precision, recall, and F1 scores before applying VIF. However, after applying VIF, XG Boost still has the highest scores, with a slight decrease in accuracy and recall but a slight increase in precision and F1 score. This indicates that XG Boost is a robust model that is less affected by multicollinearity.
On the other hand, the Naive Bayes model had the lowest scores before and after applying VIF, indicating that it is not an appropriate model for the given dataset. Furthermore, the SVM model also had low scores, indicating that it is not as effective in classifying the dataset.
In conclusion, the XG Boost and Random Forest models appear to be the most suitable for binary classification for this dataset. However, it is important to note that the effectiveness of the models can be affected by other factors, such as the quality and quantity of data, feature selection, and hyperparameter tuning.
 

Feature Importance

The most important feature for the model is "national_ inv", which has an importance score of 0.220248. The next most important features are "forecast_3_month" and "forecast_6_month" with importance scores of 0.112131 and 0.079453, respectively.
Other features that are relatively important include "sales_1_month", "sales_3_month", and "forecast_9_month". The least important features are "potential_ issue", "rev_ stop", and " oe _ constraint", all with importance scores less than 0.0001.
It's important to note that feature importance scores can vary depending on the specific model and dataset used. Therefore, it's always a good practice to evaluate the feature importance of a model to better understand which features are driving its predictions.
 
Fig:Feature Importance
 

CONCLUSION

Backorder prediction is an important task for businesses to accurately forecast and manage their inventory. By using historical data and machine learning models, it is possible to predict the likelihood of a product going on backorder and take proactive measures to prevent it from happening. This can include ordering more inventory, adjusting production schedules, or finding alternative suppliers.

The binary classification models were evaluated before and after applying the VIF technique to handle multicollinearity in the dataset. XG Boost and Random Forest were found to be the best performing models before applying VIF, and XG Boost remained the best-performing model even after applying VIF. This indicates that XG Boost is a robust model that is less affected by multicollinearity.

On the other hand, Naive Bayes and SVM models had the lowest scores before and after applying VIF, indicating that they may not be the best choice for this dataset. It is important to note that the effectiveness of the models can be affected by other factors, such as the quality and quantity of data, feature selection, and hyperparameter tuning. Therefore, further analysis and experimentation may be necessary to select the best model for the given dataset. Nonetheless, the results provide meaningful insights into the suitability of different models for binary classification tasks and highlight the importance of handling multicollinearity in the dataset.
 

REFERENCES

1.	Carbonneau R, Vahidov R, Laframboise K. Machine learning-Based Demand forecasting in supply chains. Int J Intell Inf Technol (IJIIT). 2007;3(4):40–57.

2.	Hearst MA, Susan TD, Edgar O, John P, Bernhard S. Support vector machines. In: IEEE intelligent systems and their applications. 1998. p. 18–28.

3.	Funahashi KI. On the approximate realization of continuous mappings by neural networks. Neural Netw. 1989;2(3):183–92.

4.	Carbonneau R, Laframboise K, Vahidov R. Application of machine learning techniques for supply chain demand forecasting. Eur J Oper Res. 2008;184(3):1140–54.

5.	Guanghui WANG. Demand forecasting of supply chain based on support vector regression method. Procedia Eng. 2012;29:280–4.

6.	Chen S, Cowan CF, Grant PM. Orthogonal least squares learning algorithm for radial basis function networks. IEEE Trans Neural Netw. 1991;2(2):302–9.

7.	Shin K, Shin Y, Kwon JH, Kang SH. Development of risk based dynamic backorder replenishment planning framework using Bayesian Belief Network. Comput Ind Eng. 2012;62(3):716–25.

8.	Acar Y, Gardner ES Jr. Forecasting method selection in a global supply chain. Int J Forecast. 2012;28(4):842–8.

9.	de Santis RB, de Aguiar EP, Goliatt L. Predicting material backorders in inventory management using machine learning. In 2017 IEEE Latin American Conference on Computational Intelligence (LA-CCI). 2017. p. 1–6.

10.	Prak D, Teunter R. A general method for addressing forecasting uncertainty in inventory models. Int J Forecast. 2019;35(1):224–38.
