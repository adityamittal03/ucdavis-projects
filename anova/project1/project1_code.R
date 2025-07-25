knitr::opts_chunk$set(echo = FALSE,
                      message = FALSE,
                      warning = FALSE,
                      fig.width= 5,
                      fig.height= 3,
                      fig.align= 'center')

# read dataset sparrow.csv
sparrow <- read.csv("~/Desktop/Year1/Spring Quarter 2022/STA106/Project1/sparrow (1).csv")

# setup basic functions for analysis
# power function
give.me.power = function(ybar,ni,MSE,alpha){
  a = length(ybar)
  nt = sum(ni)
  overall.mean = sum(ni*ybar)/nt
  phi = (1/sqrt(MSE))*sqrt( sum(ni*(ybar - overall.mean)^2)/a)
  phi.star = a *phi^2
  Fc = qf(1-alpha,a-1,nt-a)
  power = 1 - pf(Fc, a-1, nt-a, phi.star)
  return(power)
}

# confidence interval function
give.me.CI = function(ybar,ni,ci,MSE,multiplier){
  if(sum(ci) != 0 & sum(ci !=0 ) != 1){
    return("Error - you did not input a valid contrast")
  } else if(length(ci) != length(ni)){
    return("Error - not enough contrasts given")
  } else {
    estimate = sum(ybar*ci)
    SE = sqrt(MSE*sum(ci^2/ni))
    CI = estimate + c(-1,1)*multiplier*SE
    result = c(estimate,CI)
    names(result) = c("Estimate","Lower Bound","Upper Bound")
    return(result)
  }
}

# Part II)
# Summary Statistics
group.means = by(sparrow$Weight,sparrow$Treatment,mean)
group.sds   = by(sparrow$Weight,sparrow$Treatment,sd)
group.nis   = by(sparrow$Weight,sparrow$Treatment,length)
the.summary = rbind(group.means,group.sds,group.nis)
the.summary = round(the.summary,digits = 4)
colnames(the.summary) = names(group.means)
rownames(the.summary) = c("Means","Std. Dev","Sample Size")

# group means plot/table
plot(group.means, xaxt = "n", pch = 19, col = "purple",
     xlab = "Nest Group", ylab = "Weight Scale",
     main = "Average weight scale by group", type = "b")
axis(1,1:length(group.means),names(group.means))

# histogram
library(ggplot2)
ggplot(sparrow, aes(x = Weight)) +
  geom_histogram(binwidth = 2, , color = "black", fill = "white") +
  facet_grid(Treatment ~ .) +
  ggtitle("Weight by Treatment Group")

# boxplot
boxplot(sparrow$Weight ~ sparrow$Treatment,
        main = "Weight scale by group",
        ylab = "Weight Scale")

# Part III)
# finding outliers via Semi-studentized/standardized residuals
the.model = lm(Weight ~ Treatment, data = sparrow)
sparrow$ei = the.model$residuals
nt = nrow(sparrow)
a = length(unique(sparrow$Treatment))
SSE = sum(sparrow$ei^2)
MSE = SSE/(nt-a)
eij.star = the.model$residuals/sqrt(MSE)
alpha = 0.05
t.cutoff = qt(1-alpha/(2*nt), nt-a)
CO.eij = which(abs(eij.star) > t.cutoff)

# finding outliers via studentized residuals
rij = rstandard(the.model)
CO.rij = which(abs(rij) > t.cutoff)

# remove outlier row
CO1 = c(CO.rij)
outliers = CO1
new.data = sparrow[-outliers,]
new.model = lm(Weight ~ Treatment, data = new.data)

# assessing normality using qq plot
qqnorm(new.model$residuals)
qqline(new.model$residuals)

# assessing normality Shapiro Wilkis test
ei = new.model$residuals
the.SWtest = shapiro.test(ei)

# Assess constant variance
plot(new.model$fitted.values, new.model$residuals,
     main = "Errors vs. Group Means",
     xlab = "Group Means", ylab = "Errors", pch = 19)
abline(h = 0, col = "red")

# Brown-Forsythe Test
library(car)
the.BFtest = leveneTest(ei ~ Treatment, data = new.data, center=median)
p.val = the.BFtest[[3]][1]

# anova table
new.model = lm(Weight ~ Treatment, data = new.data)
anova.table = anova(new.model)
anova.table

# get power of test
the.power = give.me.power(group.means,group.nis,MSE,0.05)

# Confidence intervals
t.value = qt(1-0.05/2, sum(group.nis) - length(group.nis))
ci.1 = c(1,0,0)
ci.2 = c(0,1,0)
ci.3 = c(0,0,1)
CI1 = give.me.CI(group.means,group.nis,ci.1,MSE,t.value)
CI2 = give.me.CI(group.means,group.nis,ci.2,MSE,t.value)
CI3 = give.me.CI(group.means,group.nis,ci.3,MSE,t.value)

# pairwise CI
ci.4 = c(1,0,-1)  # control - reduced
ci.5 = c(1,-1,0)  # control - enlarged
CI4 = give.me.CI(group.means,group.nis,ci.4,MSE,t.value)
CI5 = give.me.CI(group.means,group.nis,ci.5,MSE,t.value)
