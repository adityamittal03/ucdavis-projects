knitr::opts_chunk$set(warning = FALSE, message = FALSE)

# PROBLEM I
# read dataset
Helicopter <- read.csv("~/Desktop/Year1/Spring Quarter 2022/STA106/Project II/Helicopter.csv")

# diagnostics
model.helicopter = lm(Count~Shift, data = Helicopter)
qqnorm(model.helicopter$residuals)
qqline(model.helicopter$residuals)
plot(model.helicopter$fitted.values, model.helicopter$residuals,
     main = "Errors vs. Group Means", xlab = abline(h = 0, col = "red"))
helicopter.original.ei = model.helicopter$residuals
helicopter.original.the.SWtest = shapiro.test(helicopter.original.ei)

library(car)
the.BFtest = leveneTest(helicopter.original.ei ~ Shift, data = Helicopter, center = median)
helicopter.original.p.val = the.BFtest[[3]][1]

# Outliers
Helicopter$ei = model.helicopter$residuals
Helicopter.nt = nrow(Helicopter)
Helicopter.a = length(unique(Helicopter$Shift))
t.cutoff= qt(1-0.01, Helicopter.nt - Helicopter.a)
Helicopter.rij = rstandard(model.helicopter)
Helicopter.CO.rij = which(abs(Helicopter.rij) > t.cutoff)
Helicopter.outliers = Helicopter.CO.rij
Helicopter.new.data = Helicopter[-Helicopter.outliers,]
Helicopter.new.model = lm(Count ~ Shift, data = Helicopter.new.data)

qqnorm(Helicopter.new.model$residuals)
qqline(Helicopter.new.model$residuals)
plot(Helicopter.new.model$fitted.values, Helicopter.new.model$residuals,
     main = "Errors vs. Group Means", abline(h = 0, col = "red"))
helicopter.outlier.ei = Helicopter.new.model$residuals
helicopter.outlier.the.SWtest = shapiro.test(helicopter.outlier.ei)

library(car)
helicopter.outlier.the.BFtest = leveneTest(helicopter.outlier.ei ~ Shift,
                                           data = Helicopter.new.data, center = median)
helicopter.outlier.p.val = helicopter.outlier.the.BFtest[[3]][1]

# BOX COX QQ Plot
library(EnvStats)
L1 = boxcox(Helicopter.new.model, objective.name = "PPCC", optimize = TRUE)$lambda
QQ.YT = (Helicopter.new.data$Count^L1 - 1)/L1
QQ.t.data = data.frame(Count = QQ.YT, Shift = Helicopter.new.data$Shift)
QQ.t.model = lm(Count ~ Shift, data = QQ.t.data)
qqnorm(QQ.t.model$residuals)
qqline(QQ.t.model$residuals)
plot(QQ.t.model$fitted.values, QQ.t.model$residuals,
     main = "Errors vs. Group Means", xlab = "Group Means", abline(h = 0, col = "orange"))
QQ.ei = QQ.t.model$residuals
QQ.the.SWtest = shapiro.test(QQ.ei)
QQ.the.BFtest = leveneTest(QQ.ei ~ Shift, data = Helicopter.new.data, center = median)
QQ.p.val = QQ.the.BFtest[[3]][1]

# BOX COX SHAPIRO
L2 = boxcox(Helicopter.new.model, objective.name = "Shapiro-Wilk", optimize = TRUE)$lambda
Shapiro.YT = (Helicopter.new.data$Count^L2 - 1)/L2
Shapiro.t.data = data.frame(Count = Shapiro.YT, Shift = Helicopter.new.data$Shift)
Shapiro.t.model = lm(Count ~ Shift, data = Shapiro.t.data)
qqnorm(Shapiro.t.model$residuals)
qqline(Shapiro.t.model$residuals)
plot(Shapiro.t.model$fitted.values, Shapiro.t.model$residuals,
     main = "Errors vs. Group Means", abline(h = 0, col = "black"))
Shapiro.ei = Shapiro.t.model$residuals
Shapiro.the.SWtest = shapiro.test(Shapiro.ei)
Shapiro.the.BFtest = leveneTest(Shapiro.ei ~ Shift, data = Helicopter.new.data, center = median)
Shapiro.p.val = Shapiro.the.BFtest[[3]][1]

# BOX COX LOG LIKELIHOOD
L3 = boxcox(Helicopter.new.data$Count, objective.name = "Log-Likelihood", optimize = TRUE)$lambda
Log.YT = (Helicopter.new.data$Count^L3 - 1)/L3
Log.t.data = data.frame(Count = Log.YT, Shift = Helicopter.new.data$Shift)
Log.t.model = lm(Count ~ Shift, data = Log.t.data)
qqnorm(Log.t.model$residuals)
qqline(Log.t.model$residuals)
plot(Log.t.model$fitted.values, Log.t.model$residuals,
     main = "Errors vs. Group Means", abline(h = 0, col = "black"))
Log.ei = Log.t.model$residuals
Log.the.SWtest = shapiro.test(Log.ei)
Log.the.BFtest = leveneTest(Log.ei ~ Shift, data = Helicopter.new.data, center = median)
Log.p.val = Log.the.BFtest[[3]][1]

knitr::include_graphics("/Users/adityamittal/Desktop/Year1/Spring Quarter 2022/STA106/Project II/Screen ...")

# setup basic functions
find.means = function(the.data, fun.name = mean){
  a = length(unique(the.data[,2]))
  b = length(unique(the.data[,3]))
  means.A = by(the.data[,1], the.data[,2], fun.name)
  means.B = by(the.data[,1], the.data[,3], fun.name)
  means.AB = by(the.data[,1], list(the.data[,2], the.data[,3]), fun.name)
  MAB = matrix(means.AB, nrow = b, ncol = a, byrow = TRUE)
  colnames(MAB) = names(means.A)
  rownames(MAB) = names(means.B)
  MA = as.numeric(means.A); names(MA) = names(means.A)
  MB = as.numeric(means.B); names(MB) = names(means.B)
  MAB = t(MAB)
  results = list(A = MA, B = MB, AB = MAB)
  return(results)
}

Partial.R2 = function(small.model, big.model){
  SSE1 = sum(small.model$residuals^2)
  SSE2 = sum(big.model$residuals^2)
  PR2 = (SSE1 - SSE2)/SSE1
  return(PR2)
}

find.mult = function(alpha, a, b, dfSSE, g, group){
  if(group == "A"){
    Tuk = round(qtukey(1-alpha, a, dfSSE)/sqrt(2),3)
    Bon = round(qt(1-alpha/(2*g), dfSSE),3)
    Sch = round(sqrt((a-1)*qf(1-alpha, a-1, dfSSE)),3)
  } else if(group == "B"){
    Tuk = round(qtukey(1-alpha, b, dfSSE)/sqrt(2),3)
    Bon = round(qt(1-alpha/(2*g), dfSSE),3)
    Sch = round(sqrt((b-1)*qf(1-alpha, b-1, dfSSE)),3)
  } else if(group == "AB"){
    Tuk = round(qtukey(1-alpha, a*b, dfSSE)/sqrt(2),3)
    Bon = round(qt(1-alpha/(2*g), dfSSE),3)
    Sch = round(sqrt((a*b-1)*qf(1-alpha, a*b-1, dfSSE)),3)
  }
  results = c(Bon, Tuk, Sch)
  names(results) = c("Bonferroni","Tukey","Scheffe")
  return(results)
}

scary.CI = function(the.data, MSE, equal.weights = TRUE, multiplier, group, cs){
  if(sum(cs) != 0 & sum(cs != 0) != 1){
    return("Error - you did not input a valid contrast")
  } else {
    the.means = find.means(the.data)
    the.ns    = find.means(the.data, length)
    nt = nrow(the.data)
    a = length(unique(the.data[,2]))
    b = length(unique(the.data[,3]))
    if(group == "A"){
      if(equal.weights){
        a.means = rowMeans(the.means$AB)
        est     = sum(a.means * cs)
        mul     = rowSums(1/the.ns$AB)
        SE      = sqrt(MSE/b^2 * sum(cs^2 * mul))
        N       = names(a.means)[cs != 0]
        CS      = paste("(", cs[cs != 0], ")", sep = "")
        fancy   = paste(paste(CS, N, sep = ""), collapse = "+")
        names(est) = fancy
      } else {
        a.means = the.means$A
        est     = sum(a.means * cs)
        SE      = sqrt(MSE * sum(cs^2 * (1/the.ns$A)))
        N       = names(a.means)[cs != 0]
        CS      = paste("(", cs[cs != 0], ")", sep = "")
        fancy   = paste(paste(CS, N, sep = ""), collapse = "+")
        names(est) = fancy
      }
    } else if(group == "B"){
      if(equal.weights){
        b.means = colMeans(the.means$AB)
        est     = sum(b.means * cs)
        mul     = colSums(1/the.ns$AB)
        SE      = sqrt(MSE/a^2 * sum(cs^2 * mul))
        N       = names(b.means)[cs != 0]
        CS      = paste("(", cs[cs != 0], ")", sep = "")
        fancy   = paste(paste(CS, N, sep = ""), collapse = "+")
        names(est) = fancy
      } else {
        b.means = the.means$B
        est     = sum(b.means * cs)
        SE      = sqrt(MSE * sum(cs^2 * (1/the.ns$B)))
        N       = names(b.means)[cs != 0]
        CS      = paste("(", cs[cs != 0], ")", sep = "")
        fancy   = paste(paste(CS, N, sep = ""), collapse = "+")
        names(est) = fancy
      }
    } else if(group == "AB"){
      est = sum(cs * the.means$AB)
      SE  = sqrt(MSE * sum(cs^2 / the.ns$AB))
      names(est) = "someAB"
    }
    the.CI  = est + c(-1,1) * multiplier * SE
    results = c(est, the.CI)
    names(results) = c(names(est), "lower bound", "upper bound")
    return(results)
  }
}

# PROBLEM II
# read dataset Salary.csv
Salary <- read.csv("~/Desktop/Year1/Spring Quarter 2022/STA106/Project II/Salary.csv")

# Summary data
Salary.the.means = find.means(Salary, mean)
Salary.the.sizes = find.means(Salary, length)
Salary.the.sds   = find.means(Salary, sd)
names(Salary)    = c("Y","A","B")
the.summary.A    = rbind(Salary.the.means$A, Salary.the.sds$A, Salary.the.sizes$A)
colnames(the.summary.A) = names(Salary.the.means$A)
rownames(the.summary.A) = c("Means","Std. Dev","Sizes")
the.summary.A

boxplot(Salary$Y ~ Salary$A, main = "Salary scale by group", ylab = "Salary Scale")
library(ggplot2)
ggplot(Salary, aes(x = Y)) +
  geom_histogram(binwidth = 2,, color = "black", fill = "white") +
  facet_grid(A ~ .) +
  ggtitle("Salary by Profession Group")

the.summary.B = rbind(Salary.the.means$B, Salary.the.sds$B, Salary.the.sizes$B)
colnames(the.summary.B) = names(Salary.the.means$B)
rownames(the.summary.B) = c("Means","Std. Dev","Sizes")
the.summary.B

boxplot(Salary$Y ~ Salary$B, main = "Salary scale by group", ylab = "Salary Scale")
ggplot(Salary, aes(x = Y)) +
  geom_histogram(binwidth = 2,, color = "black", fill = "white") +
  facet_grid(B ~ .) +
  ggtitle("Salary by Treatment Group")

Salary.the.means$AB
boxplot(Y ~ A*B, main = "Salary scale by Treatment", ylab = "Salary Scale",
        xlab = "Treatment", data = Salary)
