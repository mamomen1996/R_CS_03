# Case-Study Title: Simple Linear Regression Analysis
# Data Analysis methodology: CRISP-DM
# Dataset: Toyota Used Cars certified features and dealing (sold) prices in Europe
# Case Goal: Price Recommendation Intelligence System for Toyota Used Cars in Europe Trading Platform


### Required Libraries ----
install.packages('car')
install.packages('ggplot2')
install.packages('corrplot')
install.packages('moments')
library('car')
library('ggplot2')
library('corrplot')
library('moments')


### Read Data from File ----
data <- read.csv('CS_03.csv', header = T)
dim(data)  # 1325 records, 10 variables


### Step 1: Business Understanding ----
 # know business process and issues
 # know the context of the problem
 # know the order of numbers in the business


### Step 2: Data Understanding ----
### Step 2.1: Data Inspection (Data Understanding from Free Perspective) ----
## Dataset variables definition
colnames(data)

#variable name     what does measure
#"Price"         : Sales (sold) price in Euro      -> what we want to predict
#"Age"           : Age of a used car in month 
#"KM"            : Kilometerage usage
#"FuelType"      : Petrol, Diesel, CNG             -> Categorical (factor)
#"HP"            : Horse power     
#"MetColor"      : 1 : if Metallic color, 0 : Not  -> Categorical (factor)
#"Automatic"     : 1 : if Automatic, 0 : Not       -> Categorical (factor)
#"CC"            : Engine displacement in cc
#"Doors"         : # of doors                      -> Categorical (factor)
#"Weight"        : Weight in Kilogram


### Step 2.2: Data Exploring (Data Understanding from Statistical Perspective) ----
## Overview of Dataframe
class(data)
head(data)
tail(data)
str(data)
summary(data)

## Categorical variables should be stored as factor
cat_var <- c('FuelType', 'MetColor', 'Automatic', 'Doors')
data[, cat_var] <- lapply(data[, cat_var], factor)

summary(data)
#we have few data|sample in 'CNG' category of 'FuelType' -> it can affect on price prediction of this cars category
#we have few data|sample in '1' category of 'Automatic' -> it can affect on price prediction of this cars category

## Univariate Profiling (check each variable individually)
# Categorical variables
#check to sure that have good car distribution in each category
table(data$FuelType)  # CNG cars sample size is very small -> 17/nrow(data) < 0.05
table(data$MetColor)
table(data$Automatic)
table(data$Doors)  # 2-door cars sample size is very small -> 2/nrow(data) < 0.05

data[data$Doors == 2,]  # abnormality (error in data recording process)

# Continuous variables
par(mfrow = c(2, 3))
for(i in c(1, 2, 3, 5, 8, 10)){
	hist(data[,i], xlab='', main=paste('Histogram of', colnames(data)[i]))  # plot 6 histogram of continuous variables in one chart
}  # 'Price' is skewed to right

par(mfrow = c(1, 1))

boxplot(data$Price, main='Price Distribution')  # outlier detection by Tukey method in Price

## Bivariate Profiling (measure 2-2 relationships between variables)
# Two Continuous variables (Correlation Analysis)
cor(data$Price, data$KM)  # high correlation for this context (Used Car price)
cor_table <- round(cor(data[,c(1, 2, 3, 5, 8, 10)]), 2)  # correlation table between price and continuous variables
cor_table   # choose continuous variables which have high corr with price and consider them as feature in regression model (which variable is important for price prediction)
corrplot(cor_table)  # 'CC' has very small corr with 'price', so it can not be good predictor in modeling

#Multicollinearity (having high correlation between predictor variables):
#abs(corr) >= 0.30: Multicollinearity problem danger!
# 'Weight' has 0.66 corr with 'CC'
# 'KM' has 0.39 corr with 'Age'
# 'KM' has 0.39 corr with 'CC'
# 'KM' has -0.33 corr with 'HP'

#Scatter Plot (between price and other continuous variables 2 by 2)
par(mfrow = c(2, 3))
for(i in c(2, 3, 5, 8, 10)){
	plot(data[,i], data$Price, xlab='', main=paste('Price vs.', colnames(data)[i]))
}
par(mfrow = c(1, 1))


### Step 3: Data PreProcessing ----
# Divide Dataset into Train and Test randomly
#learn model in Train dataset
#evaluate model performance in Test dataset
set.seed(123456)
train_cases <- sample(1:nrow(data), nrow(data) * 0.7)  # according to the dataset size: 70% - 30% 
train <- data[train_cases, ]
test <- data[-train_cases, ]

#train data distribution must be similar to test data distribution
dim(train)
summary(train)
dim(test)
summary(test)


### Step 4: Modeling ----
# Train Simple Linear Regression Model 1 (Univariate Regression)
#based-on previous analysis, it seems that KM is important to explain Price variance (corr = -52%)

#Regress Price on KM
m1 <- lm(Price ~ KM, data = train)
m1  # regression equation
summary(m1)  # results of m1 regression model
#R-squared = 0.2624: 26% of 'Price' variance has been explained by 'KM'
#consider the problem context, for price prediction, R-squared = 0.26 is not good model, we need 0.70

ggplot(train, aes(x=KM, y=Price)) +
	geom_point() +
	geom_smooth(method='lm', se=F)  # variance of 'Price' based-on 'KM' is high around regression line

#Main Question: can we generalize this line to population? -> F-test and then t-test
#Check Assumptions of Regression
#1. Normality of residuals (Errors)
m1$residuals

hist(m1$residuals, probability = T)  # skewed to right (have a tail along right)
lines(density(m1$residuals), col='red')

qqnorm(m1$residuals, main='QQ Plot of residuals', pch = 20)  # we have serious deviations from normal distribution
qqline(m1$residuals, col='red')

jarque.test(m1$residuals)
#p-value < 0.05 reject normality assumption

anscombe.test(m1$residuals)
#p-value < 0.05 reject normality assumption

#result: Residuals are not Normally Distributed -> reject first Assumption of Regression

#2. Residuals independency
plot(m1)  # Diagnostic Plots

#result: We see Heteroscedasticity problem in model

plot(data$KM, data$Price)  # it seems that the relationship between these two variable in this sample and this data-range isn't linear, it is non-linear relationship 
#so, this model has problem. and t-test results of it are not reliable yet!

# Train Simple Linear Regression Model 2 (Multivariate Regression)
m2 <- lm(Price ~ KM + I(KM^2), data = train)  # Quadratic Regression
summary(m2)

#Check Assumptions of Regression
#1. Normality of residuals (Errors)
hist(m2$residuals, freq = F)
lines(density(m2$residuals), col = 'red')  # right skewed

qqnorm(m2$residuals, main = 'QQ Plot of residuals', pch = 20)
qqline(m2$residuals, col = 'red')

jarque.test(m2$residuals)
#p-value < 0.05 reject normality assumption

anscombe.test(m2$residuals)
#p-value < 0.05 reject normality assumption

#result: Residuals are not Normally Distributed -> reject first Assumption of Regression

#2. Residuals independency
plot(m2)  # Diagnostic Plots

#Linear vs. Quadratic Regression
ggplot(train, aes(KM, Price)) +
	geom_point() +
	geom_smooth(method = 'lm', formula = y ~ x + I(x^2)) +
	geom_smooth(method = 'lm', formula = y ~ x, color = 'red') +
	ggtitle('Price vs. KM')

#back to errors non-normality problem: maybe some outlier observations make them non-normal
#(in other words: residuals have not systematic deviations from normal distribution and just some outliers make them non-normal)
plot(m2)
train2 <- train[-which(rownames(train) == 7 | rownames(train) == 32 | rownames(train) == 1325),]  # remove three outliers (based-on residuals QQ-plot) from dataset

# Model 2_2
m2_2 <- lm(Price ~ KM + I(KM^2), data = train2)
summary(m2_2)

#Normality of residuals test
hist(m2_2$residuals, probability = TRUE)
lines(density(m2_2$residuals), col = "red")  # skewed to right

qqnorm(m2_2$residuals, main = "QQ Plot of residuals", pch = 20)
qqline(m2_2$residuals, col = "red")

jarque.test(m2_2$residuals)
#p-value < 0.05 reject normality assumption

anscombe.test(m2_2$residuals)
#p-value < 0.05 reject normality assumption

#result: errors are not normally distributed

#Note: Deviation from normality assumption got improved (because we remove 3 outliers)
#so, maybe Deviation from normality is just because some observations (and if remove more outliers, it will become normal)

plot(m2_2)  # Diagnostic Plots

#How we measure|detect multicollinearity? use VIF index
#If VIF > 10 then multicollinearity problem is high (big VIF is bad)
car::vif(m2)
car::vif(m2_2)

#scale 'KM' to solve VIF
train$KM_scaled <- scale(train$KM)  # Z-Normalization
head(train)

# Model 2_3
m2_3 <- lm(Price ~ KM_scaled + I(KM_scaled^2), data = train)
summary(m2_3)
plot(m2_3)
car::vif(m2_3)

#model m2 pluralization: 
## consider KM and KM^2 (are important variables to explain 'Price': explain 30% variance of 'Price')
## improve R^2
## solve Heteroscedasticity problem
## Regression Normality Assumption violation did not solve -> maybe is just because some outliers (if be true, we can remove them and solve this problem)
## we use KM_scaled and KM_scaled^2 to prevent Multicollinearity problem

# Model 3: we want to bring all variables of dataset to model
summary(train[,c(4,6,7,9)])
train$FuelType <- factor(train$FuelType, levels = c("Petrol", "Diesel", "CNG"))
train$MetColor <- factor(train$MetColor, levels = c("1", "0"))
train$Automatic <- factor(train$Automatic, levels = c("0", "1"))
train$Doors <- factor(train$Doors, levels = c("3", "5", "4", "2"))

m3 <- lm(Price ~ KM_scaled + I(KM_scaled^2) + Age + FuelType + HP + MetColor + Automatic + CC + Doors + Weight, data = train)
summary(m3)

plot(m3)
#1- Heteroscedasticity -> we have a little problem
#2- QQ-plot -> serious skewness to left
#3- like 1
#4- we have one observation in Cook's Distance zone (has huge effect on our model)

# Model 3_2: revise m3 model with significant variables to predict 'Price':
m3_2 <- lm(Price ~ KM_scaled + I(KM_scaled^2) + Age + FuelType + Automatic + CC + Weight + Doors, data = train)
summary(m3_2)
plot(m3_2)
car::vif(m3_2)

# Model 4
#create another variable:
train$IfPetrol <- ifelse(train$FuelType == 'Petrol', T, F)  # put the Petrol as the base category
head(train)
summary(train$IfPetrol)  # solve CNG small sample size problem and make balance between categories (Capping)

#remove Doors from model (does not have significant effect on Price according to t-test results)
#put IfPetrol instead of FuelType
#remove HP
#remove MetColor (p-value is too high)
m4 <- lm(Price ~ KM_scaled + I(KM_scaled^2) + Age + IfPetrol + Automatic + CC + Weight, data = train)

summary(m4)
#Adjusted R-squared did not change -> confirm that the removed variables were not important variables for explanation of Price
#F-test: p-value < 0.05 -> there is at least one linear relationship between one of the predictor variables and Price

#Check Assumptions of Regression
#1. Normality of residuals
hist(m4$residuals, freq = F, breaks = 25)
lines(density(m4$residuals), col='red')  # a little skewed to left

qqnorm(m4$residuals, main='QQ Plot of residuals', pch = 20)
qqline(m4$residuals, col = 'red')

jarque.test(m4$residuals)
#p-value < 0.05 reject normality assumption

anscombe.test(m4$residuals)
#p-value < 0.05 reject normality assumption

#result: Residuals are not Normally Distributed! (normality assumption did not confirm)

#Diagnostic Plots
plot(m4)
#1- Hetersocedasticity: a little
#2- QQ-plot
#3- like 1
#4- two outliers based-on Cook's Distance (if we remove them, will have important change on Regression line)

# Model 4_2: remove some outlier observations to improve model (to close model errors distribution to Normal)
# maybe just some outlier observations are cause of Errors Normality Assumption violation. 

plot(m4)
#remove observations 112 and 491 and 850 (have high Leverage on Regression Line based-on Cook's Distance)
#remove observations 491 and 284 and 283 (based-on QQ-plot)

train2 <- train[-which(rownames(train) %in% c(112, 284, 293, 491, 850)), ]  # iteration 1) remove some outliers based-on Diagnostic Plots

m4_2 <- lm(Price ~ KM_scaled + I(KM_scaled^2) + Age + IfPetrol + Automatic + CC + Weight, data = train2)
summary(m4_2)
#R-squared improved
#KM^2 marginal significant
#Automatic1 is not significant
#CC is not significant

plot(m4_2)

train2 <- train[-which(rownames(train) %in% c(112, 114, 284, 293, 491, 544, 850, 948, 1325)),]  # iteration 2) remove some outliers based-on Diagnostic Plots
m4_2 <- lm(Price ~ KM_scaled + I(KM_scaled^2) + Age + IfPetrol + Automatic + CC + Weight, data = train2)
summary(m4_2)
plot(m4_2)

train2 <- train[-which(rownames(train) %in% c(77, 112, 114, 284, 293, 491, 544, 586, 803, 850, 881, 944, 948, 1241, 1325)), ] #iteration 3) remove some outliers based-on Diagnostic Plots
m4_2 <- lm(Price ~ KM_scaled + I(KM_scaled^2) + Age + IfPetrol + Automatic + CC + Weight, data = train2)
summary(m4_2)
#Automatic1 is not significant (has high p-value)

plot(m4_2)

# Model 4_3: remove Automatic
m4_3 <- lm(Price ~ KM_scaled + I(KM_scaled^2) + Age + CC + IfPetrol + Weight, data = train2)
summary(m4_3)
#F-test: p-value < 0.05
#R-squared: 83%
#t-test: all variables are significant

#Normality of residuals (Check Normality Assumption)
hist(m4_3$residuals, breaks = 20, freq = F)
lines(density(m4_3$residuals), col = 'red')

qqnorm(m4_3$residuals, main='QQ Plot of residuals', pch = 20)
qqline(m4_3$residuals, col='red')

jarque.test(m4_3$residuals)
#p-value < 0.05 reject normality assumption

anscombe.test(m4_3$residuals)
#p-value < 0.05 reject normality assumption

#result: residuals are Normal

#Diagnostic plots
plot(m4_3)  # all are good

#Check Multicollinearity
car::vif(m4_3)  # have no value over 10 (or over 5) -> there isn't Mulitcollinearity problem

#Compare Two models
coef(m4)  # model with outliers
coef(m4_3)  # model without outliers

#number of removed cases:
dim(train)
dim(train2)
nrow(train)-nrow(train2)  # 15 Observations
(nrow(train)-nrow(train2))/nrow(train)*100  # 1.6% of observations is removed


### Step 5: Model Evaluation ----
# Test the Model 4_3 performance

#Data Preparation
head(test)
test$KM_scaled <- scale(test$KM)
test$IfPetrol <- ifelse(test$FuelType == 'Petrol', T, F)
head(test)
dim(test)

#Prediction
test$pred <- predict(m4_3, test)  # input test data into m4_3 model (use model m4_3 to predict Price on Test dataset)
head(test)

#Evaluate model performance in Test dataset:
#Actual vs. Prediction
plot(x = test$Price, y = test$pred, xlab = 'Actual', ylab = 'Prediction')
abline(a = 0, b = 1, col = 'red', lwd = 3)  # compare with 45' line

#Absolute Error mean, median, sd, max, min
abs_err <- abs(test$Price - test$pred) #absolute value (AEV)

hist(abs_err, breaks = 25)  # residuals distribution
mean(abs_err)
median(abs_err)
sd(abs_err)
max(abs_err)
min(abs_err)

#boxplot (which observations are outliers?)
boxplot(abs_err, main = 'Error distribution')

#Error Percentage median, sd, mean, max, min
e_percent <- abs(test$Price - test$pred)/test$Price * 100
summary(e_percent)

sum(e_percent <= 15)/length(e_percent)*100  # 84.17% of predicted prices fall between [-15%, +15%] actual price

#we created a robust model
