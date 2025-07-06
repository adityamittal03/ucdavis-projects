### Final Project ECS_189G
### adimittal@ucdavis.edu
### R file

library("Kendall")
library("regtools")
library("ggplot2")
library('qeML')

load('/Users/adityamittal/Desktop/Year_two/Spring_2023/ECS_189G/packages/fairml/data/german.credit.rda') # load data
View(german.credit)
str(german.credit)

# Exploratory Data Analysis ----------------------------------------------------------------------

# Distribution of our response variable "Credit_risk"
ggplot(data=german.credit, aes(x=Credit_risk)) + geom_bar() + 
  geom_text(stat='count', aes(label=..count..), vjust=1.6, color="white") + 
  ggtitle("Proportions of Bad vs. Good Credit Risk") + xlab("Credit Risk") + ylab("Frequency Counts")

# drop foreign worker
drop <- c("Foreign_worker") 
new_df = german.credit[,!(names(german.credit) %in% drop)]

# Finding potential proxies ----------------------------------------------------------------------

# our proxy is "Present_employment_since"

# proportion of males vs females across different levels of employment status
View(prop.table(table(new_df$Gender, new_df$Present_employment_since), margin = 1))

# P(Y = 1| A = a)
data_split = split(new_df,new_df[['Present_employment_since']])
a <- nrow(data_split$unemployed[data_split$unemployed['Credit_risk'] =="GOOD",])/nrow(data_split$unemployed) # 0.6290323
b <- nrow(data_split$`< 1 year`[data_split$`< 1 year`['Credit_risk'] =="GOOD",])/nrow(data_split$`< 1 year`) # 0.5930233
c <- nrow(data_split$`1  <= ... < 4 years`[data_split$`1  <= ... < 4 years`['Credit_risk'] =="GOOD",])/nrow(data_split$`1  <= ... < 4 years`) # 0.6932153
d <- nrow(data_split$`4  <= ... < 7 years`[data_split$`4  <= ... < 7 years`['Credit_risk'] =="GOOD",])/nrow(data_split$`4  <= ... < 7 years`) # 0.7758621
e <- nrow(data_split$`>= 7 years`[data_split$`>= 7 years`['Credit_risk'] =="GOOD",])/nrow(data_split$`>= 7 years`) # 0.7470356

db <- data.frame ('unemployed' = a,
                  '< 1 year'  = b,
                  '1  <= ... < 4 years' = c,
                  '4  <= ... < 7 years' = d,
                  '>= 7 years' = e)
View(db)

#  Explicitly Deweighted Features ------------------------------------------------------------------------------

# update dataset
new_df$Credit_risk <- ifelse(new_df$Credit_risk == "GOOD",1,2) # good is 1.
new_df$Credit_risk <- as.factor(new_df$Credit_risk)

new_df$Gender <- ifelse(new_df$Gender == "Female",1,2) # convert female to numeric
new_df$Gender <- as.factor(new_df$Gender)

qe_noS <- c("Gender", "Age")
qe_df <- new_df[,!(names(new_df) %in% qe_noS)]

# Compute testAcc of entire model
a_testAcc <- replicMeans(25,"qeKNN(new_df,'Credit_risk')$testAcc")

# Compute utility for suppression pre-processing method
b_testAcc <- replicMeans(25,"qeKNN(qe_df,'Credit_risk')$testAcc")

# one-hot encode employment levels to use in expandVars
employment_levels <- factorToDummies(new_df$Present_employment_since,'Present_employment_since', omitLast = TRUE) # create dummy variables 
df1 <- cbind(new_df,employment_levels) # create new dataframe w/ dummy variables
df1 <- df1[,!names(df1) %in% c('Present_employment_since','Gender','Age')] # remove variables
View(df1)

d = seq(0.1,1,0.1) # 0-1; intervals of 0.1 (10 values)

# compute utility and fairness
testAcc = c()  # initialize empty vectors 
mean_cor = c() 
mean_cor_2 = c()

for (i in d) {
  # utility
  val = replicMeans(25,"qeKNN(df1,'Credit_risk',expandVars=c('Present_employment_since.< 1 year','Present_employment_since.>= 7 years','Present_employment_since.1  <= ... < 4 years','Present_employment_since.4  <= ... < 7 years'), expandVals=c(i,i,i,i),scaleX = TRUE)$testAcc")
  testAcc = c(testAcc, val)
  
  # fairness
  cor_vector = c() # computes 25 correlation values across every i-th run
  cor_vector_2 = c()
  for (j in 1:25) { # this for loop is intended to run each iteration of d 25 times (like replicMeans) and compute the overall mean from all runs
    run <- qeKNN(df1,'Credit_risk',expandVars=c('Present_employment_since.< 1 year','Present_employment_since.>= 7 years','Present_employment_since.1  <= ... < 4 years','Present_employment_since.4  <= ... < 7 years'),expandVals=c(i,i,i,i),scaleX = TRUE) # run the qeKNN function j times for each i-th iteration
    index <- run$holdIdxs # get index values of holdout set 
    S_val = new_df[index,]$Gender # get S values of holdout set from original data
    S_val_2 = new_df[index,]$Age
    predicted_val = run$holdoutPreds$probs # get predicted values from holdout set
    v = Kendall(predicted_val,S_val) # compute Kendall correlation
    v2 = Kendall(predicted_val, S_val_2)
    cor_val = v$tau[1] # store correlation
    cor_val_2 = v2$tau[1]
    cor_vector = c(cor_vector, cor_val) # append cor to vector
    cor_vector_2 = c(cor_vector_2,cor_val_2)
  }
  mean_val = mean(cor_vector)
  mean_cor = c(mean_cor, mean_val)
  
  mean_val_2 = mean(cor_vector_2)
  mean_cor_2 = c(mean_cor_2, mean_val_2)
}

df_results <- data.frame (d  = d,
                          Utility = testAcc,
                          Fairness_Sex = mean_cor,
                          Fairness_Age = mean_cor_2) # create dataframe
View(df_results)

# Creating the plot for Utility and Fairness:
par(mfrow = c(1, 3))
plot(d,testAcc, type="l", col="black", lwd=1, xlab="D", ylab="Utility (Test Accuracy)")
title("Utility against D (Test Accuracy)")
plot(d,mean_cor, type="l", col="blue", lwd=1, xlab="D", ylab="Fairness (Kendall Correlation)")
title("Fairness against D (Sex)")
plot(d,mean_cor_2, type="l", col="blue", lwd=1, xlab="D", ylab="Fairness (Kendall Correlation)")
title("Fairness against D (Age)")
par(mfrow = c(1, 1))

# Compute statistical parity & disparate parity for raw model
a <- qeKNN(new_df,'Credit_risk')
temp_data_2 = new_df[,-19]
pred_2 = c()
for (k in 1:nrow(temp_data_2)) {
  y_k = predict(a,temp_data_2[k,])$predClasses
  pred_2 <- c(pred_2, y_k)
}
temp_data_2 <- cbind(temp_data_2,pred_2)

male <- temp_data_2[temp_data_2['Gender'] == 1,]
nrow(male[male['pred_2'] == 1,])/nrow(male) # 0.9434782609

fem <- temp_data_2[temp_data_2['Gender'] == 2,]
nrow(fem[fem['pred_2'] == 1,])/nrow(fem) # 0.9064516129

stat_par = (nrow(male[male['pred_2'] == 1,])/nrow(male)) - (nrow(fem[fem['pred_2'] == 1,])/nrow(fem))
dis_par = (nrow(male[male['pred_2'] == 1,])/nrow(male))/(nrow(fem[fem['pred_2'] == 1,])/nrow(fem))


# Compute statistical parity & disparate parity of EDF model
c <- qeKNN(df1,'Credit_risk',
           expandVars=c('Present_employment_since.< 1 year','Present_employment_since.>= 7 years',
                        'Present_employment_since.1  <= ... < 4 years',
                        'Present_employment_since.4  <= ... < 7 years'),expandVals=c(0.6,0.6,0.6,0.6),
           scaleX = TRUE)

temp_data = df1[,-17] 
pred = c() 

for (k in 1:nrow(temp_data)) {
  y_k = predict(c,temp_data[k,])$predClasses
  pred <- c(pred, y_k)
}
temp_data <- cbind(temp_data,pred,new_df$Gender)

male <- temp_data[temp_data['new_df$Gender'] == 1,]
nrow(male[male['pred'] == 1,])/nrow(male) # 0.94

fem <- temp_data[temp_data['new_df$Gender'] == 2,]
nrow(fem[fem['pred'] == 1,])/nrow(fem) # 0.89

stat_par2 = (nrow(male[male['pred'] == 1,])/nrow(male)) - (nrow(fem[fem['pred'] == 1,])/nrow(fem))
dis_par_2 = (nrow(male[male['pred'] == 1,])/nrow(male))/(nrow(fem[fem['pred'] == 1,])/nrow(fem))



