### ---------------------- STA 135: Final Project Code -------------------------
### ----------------------------------------------------------------------------
### Milk Transportation Costs Data
### Methods:
### One Sample Test (on gasoline)
### Two Sample Test (gasoline vs diesel sample)

# data preparation
milk_data <- read.table("~/Desktop/final_project/T6-10.dat", header = FALSE, sep = ",")
colnames(milk_data) <- c('Fuel', 'Repair', 'Capital', 'Type') # column types
milk_data[] <- lapply(milk_data, trimws) # trim white spaces
milk_data[,-4] <- lapply(milk_data[,-4], as.numeric) # convert to numeric

# split for gasoline and diesel samples
gasoline_costs <- milk_data[1:36,-4]
diesel_costs <- milk_data[37:59,-4]

###. ------------------------- One sample test ---------------------------------
# get sample statistics
x_bar <- colMeans(gasoline_costs)
Sx <- cov(gasoline_costs); Sx; dim(Sx)
n = 36
p = 3
alpha = 0.05

# linear transformation constract matrix C
C <- matrix(c(1,-1,0,
              1,0,-1), nrow = 2 , ncol = 3, byrow = TRUE); C
q = 2 # new dimension for critical value

# new summary statistics 
Sy <-  C %*% Sx %*% t(C); Sy
y_bar <- C %*% x_bar; y_bar

# test statistic
t = n * t(y_bar) %*% solve(Sy) %*% y_bar

# critical value
test_crit = (n-1)*q/(n-q) * qf(1-alpha, q, n-q)

t > test_crit # decision rule: true, reject H0

# plot mean centered-ellipse
library(ellipse)
X <- cbind(gasoline_costs[,1] - gasoline_costs[,2],
           gasoline_costs[,1] - gasoline_costs[,3])
sd <- sqrt(c(Sy[1,1], Sy[2,2]))
corr<- Sy[1,2]/sqrt(Sy[1,1]*Sy[2,2])
xbar = (1/n)*t(X)%*%rep(1,n)
plot(X, pch = 16, xlim=c(-10,10), ylim=c(-10,10), 
     main="Confidence Rigion", xlab="X1", ylab="X2")
lines(ellipse(corr, scale = sd, centre = xbar, 
              t = sqrt(test_crit/n), npoints = 25))
points(y_bar[1],y_bar[2], pch=15, col="blue")
text(y_bar[1], y_bar[2], "ybar", cex=1.3, pos=4)
points(0, 0, pch=15, col="red")
text(0, 0, "origin", cex=1.3, pos=2)


### ---------------------------- Two Sample test -------------------------------
# summary statistics for both samples
x_bar_1 <- colMeans(gasoline_costs)
S1 <- cov(gasoline_costs); S1; dim(S1)
x_bar_2 <- colMeans(diesel_costs)
S2 <- cov(diesel_costs); S2; dim(S2)
n <- c(36,23)
p = 3

# get combined result S_pooled and D
d = x_bar_1 - x_bar_2
Sp<-((n[1]-1)*S1+(n[2]-1)*S2)/(n[1] + n[2] -2)

# critical value
alpha = 0.05
test_crit = (n[1] + n[2] -2)*p/(n[1] + n[2] -p-1)*
  qf(1-alpha,p,n[1] + n[2] -p-1); test_crit

# Hotelling's T-square
Sp_norm <- (1/n[1] + 1/n[2])*Sp
T_square <- t(d)%*%solve(Sp_norm)%*%d; T_square

T_square > test_crit # decision rule: reject H0

### ----------------------------------------------------------------------------
### Abalone Data
### Methods:
### LDA for classification
### PCA for dimensionality reduc.

# data-prep
abalone_df<- read.table("~/Desktop/final_project/abalone.data", header = FALSE, sep = ",")
colnames(abalone_df) <- c('Sex', 'Length', 'Diameter', 'Height', 'Whole_Weight', 
                          'Shucked_Weight', 'Viscera_weight','Shell_weight', 
                          'Rings')

### ---------------------------- LDA -------------------------------------------
## predict gender: adults vs. infants 
abalone_df$Sex <- ifelse(abalone_df$Sex %in% c("M", "F"), "A", abalone_df$Sex)

### summary values for each group 
adult_data <- abalone_df[abalone_df$Sex == "A",-1]
infant_data <- abalone_df[abalone_df$Sex == "I",-1]
n <- c(nrow(adult_data), nrow(infant_data))

g1_bar <- colMeans(adult_data)
S1 <- cov(adult_data)
g2_bar <- colMeans(infant_data)
S2 <- cov(infant_data)
S_pooled <- ((n[1]-1)*S1+(n[2]-1)*S2)/(n[1] + n[2] -2)

# LDA decision boundary
w <- solve(S_pooled) %*% (g1_bar - g2_bar)
boundary = 0.5 * (t(w) %*% (g1_bar + g2_bar))

### Apparent error rate (training error)
preds <- c()
for (i in 1:nrow(abalone_df)) {
  row <- abalone_df[i,-1]
  val <- t(w) %*% t(as.matrix(row))
  if (val >= boundary[1,1]) {
    preds <- c(preds, "A")
  } else {
    preds <- c(preds,"I")
  }
}
sum(preds != abalone_df[,1])/nrow(abalone_df) # misclassified samples: 0.2027771

### Test Error: leave one out C.V using Lachenbruch's Holdout
preds <- c()
for (i in 1:nrow(abalone_df)) {
  print(i)
  trn_data <- abalone_df[-i,]
  tst_data <- abalone_df[i,-1]
  
  adult_data <- trn_data[trn_data$Sex == "A",-1]
  infant_data <- trn_data[trn_data$Sex == "I",-1]
  n <- c(nrow(adult_data), nrow(infant_data))
  
  g1_bar <- colMeans(adult_data)
  S1 <- cov(adult_data)
  g2_bar <- colMeans(infant_data)
  S2 <- cov(infant_data)
  S_pooled <- ((n[1]-1)*S1+(n[2]-1)*S2)/(n[1] + n[2] -2)
  
  w <- solve(S_pooled) %*% (g1_bar - g2_bar)
  boundary = 0.5 * (t(w) %*% (g1_bar + g2_bar))
  
  val <- t(w) %*% t(as.matrix(tst_data))
  if (val >= boundary[1,1]) {
    preds <- c(preds, "A")
  } else {
    preds <- c(preds,"I")
  }
}; preds

sum(preds != abalone_df[,1])/nrow(abalone_df) # misclassified samples: 0.2032559

###------------------------------ PCA ------------------------------------------
# function does PCA as by number of components
performPCA <- function(data, num_components = 1, scaleData = TRUE) {
  if (scaleData) {
    
    ### using correlation matrix R
    data <- scale(data, center = TRUE, scale = TRUE)
    covX <- cov(data)
  } else {
    
    ### using covariance matrix S
    covX <- cov(data)
  }
  
  # do spectral decomposition
  eigenDecomp <- eigen(covX)
  sorted_indices <- order(eigenDecomp$values, decreasing = TRUE)
  eigenvalues <- eigenDecomp$values[sorted_indices]
  eigenvectors <- eigenDecomp$vectors[, sorted_indices]
  
  # compute principle components
  pComponents = as.matrix(data) %*% as.matrix(eigenvectors)
  pca_scores <- pComponents[,1:num_components]
  return(list(scores = pca_scores,
              allEigen = eigenvalues,
              eigenvalues = eigenvalues[1:num_components],
              eigenvectors = eigenvectors[,1:num_components]))
}

# loss of information with regards to principle components
loss <- c()
eign <- c()
for (i in 8:1) {
  pca_result <- performPCA(abalone_df[,-1],i, scaleData = T)
  pca_scores <- pca_result$scores
  pca_eigen <- pca_result$eigenvalues
  pca_all <- pca_result$allEigen
  l = sum(pca_eigen)/sum(pca_all)
  loss = c(loss,l)
  eign <- pca_all
}; loss; eign

# plot the loss 
dd <- data.frame(x = 8:1, y = loss)
library(ggplot2)
ggplot(dd, aes(x = x, y = y)) +
  geom_line() +
  labs(x = "Number of Principle Components",
       y = "Proportion of Variance Retained") +
  ggtitle("Retention of Information vs Number of Principle Components") + 
  theme_bw()

# plot the eigenvalue vs number of principle components 
dd <- data.frame(x = 1:8, y = eign)
ggplot(dd, aes(x = x, y = y)) +
  geom_line() +
  labs(x = "Number of Principle Components",
       y = "Eigenvalue Size") +
  ggtitle("Eigenvalue Size vs Number of Principle Components") + 
  theme_bw()

# Do LDA based on n principle components - same as top
comp <- 5
pca_result_2 <- performPCA(abalone_df[,-1],comp)
pca_scores <- pca_result_2$scores
pca_eigen <- pca_result_2$eigenvalues
pca_all <- pca_result_2$allEigen
l = sum(pca_eigen)/sum(pca_all) # loss 0.92

newAba_df <- as.data.frame(cbind(abalone_df[,1], pca_scores))
colnames(newAba_df) <- c("Age", "Y1", "Y2","Y3")
newAba_df[,-1] <- lapply(newAba_df[,-1], as.numeric)

### summary values for each group 
adult_data <- newAba_df[newAba_df$Age == "A",-1]
infant_data <- newAba_df[newAba_df$Age == "I",-1]
n <- c(nrow(adult_data), nrow(infant_data))

g1_bar <- colMeans(adult_data)
S1 <- cov(adult_data)
g2_bar <- colMeans(infant_data)
S2 <- cov(infant_data)
S_pooled <- ((n[1]-1)*S1+(n[2]-1)*S2)/(n[1] + n[2] -2)

# LDA decision boundary
w <- solve(S_pooled) %*% (g1_bar - g2_bar)
boundary = 0.5 * (t(w) %*% (g1_bar + g2_bar))

### Apparent error rate (training error)
preds <- c()
for (i in 1:nrow(newAba_df)) {
  row <- newAba_df[i,-1]
  val <- t(w) %*% t(as.matrix(row))
  if (val >= boundary[1,1]) {
    preds <- c(preds, "A")
  } else {
    preds <- c(preds,"I")
  }
}
sum(preds != newAba_df[,1])/nrow(newAba_df) # misclassified samples: 0.2080441 

### Test Error: leave one out C.V using Lachenbruch's Holdout
preds <- c()
for (i in 1:nrow(newAba_df)) {
  print(i)
  trn_data <- newAba_df[-i,]
  tst_data <- newAba_df[i,-1]
  
  adult_data <- trn_data[trn_data$Age == "A",-1]
  infant_data <- trn_data[trn_data$Age == "I",-1]
  n <- c(nrow(adult_data), nrow(infant_data))
  
  g1_bar <- colMeans(adult_data)
  S1 <- cov(adult_data)
  g2_bar <- colMeans(infant_data)
  S2 <- cov(infant_data)
  S_pooled <- ((n[1]-1)*S1+(n[2]-1)*S2)/(n[1] + n[2] -2)
  
  w <- solve(S_pooled) %*% (g1_bar - g2_bar)
  boundary = 0.5 * (t(w) %*% (g1_bar + g2_bar))
  
  val <- t(w) %*% t(as.matrix(tst_data))
  if (val >= boundary[1,1]) {
    preds <- c(preds, "A")
  } else {
    preds <- c(preds,"I")
  }
}; preds

sum(preds != newAba_df[,1])/nrow(newAba_df) # misclassified samples: 0.2082835

### plots 
# plot principle components
library(ggplot2)
ggplot(newAba_df, aes(x = Y1, y = Y2, color = Age)) +
  geom_point(size = 0.5) +
  labs(x = "Y1", y = "Y2", color = "Age") +
  ggtitle("Y1 vs Y2 Scatterplot based on Age") + 
  theme_bw() + ylim(-5,2.5) + xlim(-5,5)

## visualize geometric interpretation of LDA with Y1 and Y2
mean_values <- aggregate(cbind(Y1, Y2) ~ Age, data = newAba_df, FUN = mean)
x_o <- newAba_df[1,]
a1 = as.matrix((x_o[,-1] - g1_bar)) %*% 
  solve(S_pooled) %*% t(as.matrix((x_o[,-1] - g1_bar)))
a2 = as.matrix((x_o[,-1] - g2_bar)) %*% 
  solve(S_pooled) %*% t(as.matrix((x_o[,-1] - g2_bar)))
if (a1 < a2) { # for user intuition
  print("A")
} else {
  print("I")
}
mean_values$distance <- c(round(a1,2), round(a2,2))

ggplot(newAba_df, aes(x = Y1, y = Y2, color = Age)) +
  geom_point(size = 0.5, alpha = 0.4) +
  geom_point(data = mean_values, aes(x = Y1, y = Y2), 
             color = "black", size = 2) +
  geom_text(data = mean_values[1, ], aes(label = paste("Adults")), 
            vjust = -0.5, hjust = -0.5, color = "black", fontface = "bold") +
  geom_text(data = mean_values[2, ], aes(label = paste("Infants")), 
            vjust = -0.5, hjust = -0.5, color = "black", fontface = "bold") +
  geom_point(data = data.frame(x_o), aes(x = Y1, y = Y2, color = Age), size = 3) +
  geom_segment(data = mean_values, 
               aes(x = Y1, y = Y2, xend = x_o$Y1, yend = x_o$Y2), 
               color = "purple", linetype = "dashed") +  
  geom_text(data = mean_values, 
            aes(x = (Y1 + x_o$Y1) / 2, y = (Y2 + x_o$Y2) / 2, 
                                    label = paste(distance)), 
            color = "black", hjust = 0.5, vjust = -0.5, fontface = "bold") +  
  labs(x = "Y1", y = "Y2", color = "Age Group") +
  ggtitle("Classification of xNew based on Mahalonbis Distance") +
  theme_bw() + ylim(-5,2.5) + xlim(-5, 5)
  