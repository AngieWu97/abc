#install.packages("keras")
library(keras)
library(dplyr)
library(e1071)
library(rpart)
library(caTools)
library(tree)
library(randomForest)
library(ElemStatLearn)
library(class)
library(pROC)
library(scatterplot3d)
library(rgl)
library("plot3D")
library(ggplot2)
#install.packages("pROC")

source("DataAnalyticsFunctions.R")

#Clean data has been uploaded on canvas,but we don't include the raw data because its too large. You can find it on the cms.gov.

options(warn=-1)

doc_model3<-read.csv("doc_model_final.csv")

summary(doc_model3)

# a<-doc_model[which(doc_model$entity == "individual"), ]
# summary(a)

###############

###############
set.seed(1)   
sample = sample.split(doc_model3,SplitRatio = 0.9) 
train1 =subset(doc_model3,sample ==TRUE) 
test1=subset(doc_model3, sample==FALSE)

summary(train1)
summary(test1)

###################

###################
n <- nrow(train1)
nfold <- 10
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

OOS.TPR <- data.frame(logistic=rep(NA,nfold),knn=rep(NA,nfold),tree=rep(NA,nfold),forest=rep(NA,nfold),nn=rep(NA,nfold)) 
OOS.FPR <- data.frame(logistic=rep(NA,nfold),knn=rep(NA,nfold),tree=rep(NA,nfold),forest=rep(NA,nfold),nn=rep(NA,nfold)) 
OOS.ACC <- data.frame(logistic=rep(NA,nfold),knn=rep(NA,nfold),tree=rep(NA,nfold),forest=rep(NA,nfold),nn=rep(NA,nfold)) 
OOS.AUC <- data.frame(logistic=rep(NA,nfold),knn=rep(NA,nfold),tree=rep(NA,nfold),forest=rep(NA,nfold),nn=rep(NA,nfold)) 

completedata <- train1[complete.cases(train1), ]

summary(completedata)

for(k in 1:nfold){ 
  train <- which(foldid!=k) 
  
  summary(train1[train,])
  summary(train1[-train,])
  
  x.holdout<- model.matrix(assgn ~., data=train1[-train,])[,-1]
  y.holdout<- train1[-train,]$assgn == "Y"
  
  x.data<- model.matrix(assgn ~ ., data=train1[train,])[,-1]
  y.data<- train1[train,]$assgn == "Y"
  
  #rescale (to be between 0 and 1)
  x_train <- x.data %*% diag(1/apply(x.data, 2, function(x) max(x, na.rm = TRUE)))
  y_train <- as.numeric(y.data)
  x_test <- x.holdout %*% diag(1/apply(x.data, 2, function(x) max(x, na.rm = TRUE)))
  y_test <- as.numeric(y.holdout) 
  
  #rescale (unit variance and zero mean)
  mean <- apply(x.data,2,mean)
  std <- apply(x.data,2,sd)
  x_train <- scale(x.data,center = mean, scale = std)
  y_train <- as.numeric(y.data)
  x_test <- scale(x.holdout,center = mean, scale = std)
  y_test <- as.numeric(y.holdout) 
  
  num.inputs <- ncol(x_train)
  
  model <- keras_model_sequential() %>%
    layer_dense(units=16,activation="relu",input_shape = c(num.inputs)) %>%
    layer_dense(units=32,activation="relu") %>%
    layer_dense(units=16,activation="relu") %>%
    layer_dense(units=1,activation="sigmoid")
  
  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )
  
  result<-as.numeric(train1$assgn[-train])-1
  conc<-as.numeric(train1$assgn[train])-1
  
  model.logistic <-glm(assgn~.-final_MIPS_score.x, data=train1, subset=train,family="binomial")
  pred.logistic <- predict(model.logistic, newdata=train1[-train,],type="response")
  summary(pred.logistic)
  class.log<-pred.logistic>0.95
  summary(class.log)
  class.log[class.log=="TRUE"]<-1
  class.log[class.log=="FALSE"]<-0
  summary(class.log)
  class.log.auc<-pred.logistic
  auc.log<-auc(result,class.log.auc)
  
  model.tree <- tree(assgn~.-final_MIPS_score.x, data=train1[train,]) 
  pred.tree <- predict(model.tree, newdata=train1[-train,], type="vector")
  summary(pred.tree)
  class.tree<-pred.tree[,2]>0.95
  summary(class.tree)
  class.tree[class.tree=="TRUE"]<-1
  class.tree[class.tree=="FALSE"]<-0
  summary(class.tree)
  class.tree.auc<-pred.tree[,2]
  auc.tree<-auc(result,class.tree.auc)
  
  model.forest <- randomForest(assgn~.-final_MIPS_score.x, data=train1, subset=train, nodesize=5, ntree = 500, mtry = 4)
  pred.forest <- predict(model.forest, newdata=train1[-train,],type="vote")
  summary(pred.forest)
  class.forest<-pred.forest[,2]>0.95
  summary(class.forest)
  class.forest[class.forest=="TRUE"]<-1
  class.forest[class.forest=="FALSE"]<-0
  summary(class.forest)
  class.forest.auc<-pred.forest[,2]
  auc.forest<-auc(result,class.forest.auc)
  
  model.knn <- knn(train=x_train, test=x_test, cl=y_train, k=3, prob=TRUE)
  table(model.knn,y_test)
  as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}
  mod <- as.numeric.factor(model.knn)
  auc.knn<-auc(y_test,mod)
  
  history <- model %>% fit(
    x_train, y_train, 
    epochs = 30, batch_size = 128, # choose 128 to train, train 30 times.
    validation_split = 0.3
  )
  pred.NN2 <- model%>% predict(x_test)
  class.nn <- pred.NN2>0.95
  summary(class.nn)
  class.nn[class.nn=="TRUE"] <- 1  
  class.nn[class.nn=="FALSE"] <- 0
  class.nn.auc<-pred.NN2
  auc.nn<-auc(y_test,class.nn.auc)
  
  # Logit
  value <- FPR_TPR(class.log, result)
  OOS.TPR$logistic[k] <- value$TPR
  OOS.TPR$logistic[k]
  OOS.FPR$logistic[k] <- value$FPR
  OOS.FPR$logistic[k]
  OOS.AUC$logistic[k] <- auc.log
  OOS.AUC$logistic[k]
  # Tree
  value <- FPR_TPR(class.tree, result)
  OOS.TPR$tree[k] <- value$TPR
  OOS.TPR$tree[k]
  OOS.FPR$tree[k] <- value$FPR
  OOS.FPR$tree[k]
  OOS.AUC$tree[k] <- auc.tree
  OOS.AUC$tree[k]
  # Forest
  value <- FPR_TPR(class.forest, result)
  OOS.TPR$forest[k] <- value$TPR
  OOS.TPR$forest[k]
  OOS.FPR$forest[k] <- value$FPR
  OOS.FPR$forest[k]
  OOS.AUC$forest[k] <- auc.forest
  OOS.AUC$forest[k]
  #knn
  value <- FPR_TPR(mod,y_test)
  OOS.TPR$knn[k] <- value$TPR
  OOS.TPR$knn[k]
  OOS.FPR$knn[k] <- value$FPR
  OOS.FPR$knn[k]
  OOS.AUC$knn[k] <- auc.knn
  OOS.AUC$knn[k]
  #nn
  value <- FPR_TPR(class.nn,y_test)
  OOS.TPR$nn[k] <- value$TPR
  OOS.TPR$nn[k]
  OOS.FPR$nn[k] <- value$FPR
  OOS.FPR$nn[k]
  OOS.AUC$nn[k] <- auc.nn
  OOS.AUC$nn[k]
  
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}

colMeans(OOS.TPR)
colMeans(OOS.FPR)
colMeans(OOS.AUC)
m.OOS.TPR <- as.matrix(OOS.TPR)
rownames(m.OOS.TPR) <- c(1:nfold)
barplot(t(as.matrix(OOS.TPR)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=0.5),
        ylab= bquote( "Out of Sample " ~ TPR), xlab="Fold", names.arg = c(1:10))

if (nfold >= 10){
  boxplot(OOS.TPR, col="plum", las = 2, ylab=expression(paste("OOS.TPR ",TPR)), xlab="", main="10-fold Cross Validation")
  names(OOS.TPR)[1] <-"logistic"
}
###

m.OOS.FPR <- as.matrix(OOS.FPR)
rownames(m.OOS.FPR) <- c(1:nfold)
barplot(t(as.matrix(OOS.FPR)), beside=TRUE, legend=TRUE, args.legend=c(xjust=0.5, yjust=0.3),
        ylab= bquote( "Out of Sample " ~ FPR), xlab="Fold", names.arg = c(1:10))

if (nfold >= 10){
  boxplot(OOS.FPR, col="plum", las = 2, ylab=expression(paste("OOS.FPR ",FPR)), xlab="", main="10-fold Cross Validation")
  names(OOS.FPR)[1] <-"logistic"
}
###

m.OOS.AUC <- as.matrix(OOS.AUC)
rownames(m.OOS.AUC) <- c(1:nfold)
barplot(t(as.matrix(OOS.AUC)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1.5, yjust=0.1,cex=0.8),
        ylab= bquote( "Out of Sample " ~ AUC), xlab="Fold", names.arg = c(1:10))

if (nfold >= 10){
  boxplot(OOS.AUC, col="plum", las = 2, ylab=expression(paste("OOS.TPR ",AUC)), xlab="", main="10-fold Cross Validation")
  names(OOS.AUC)[1] <-"logistic"
}

##################

model.forest <- randomForest(assgn~.-final_MIPS_score.x, data=train1, nodesize=5, ntree = 500, mtry = 4)
pred.forest <- predict(model.forest, newdata=test1,type="vote")
summary(pred.forest)
prob<-pred.forest[,2]
prob<-as.data.frame(prob)
test1
pred.test<-data.frame(test1,prob)
final_score<-pred.test$final_MIPS_score.x*pred.test$prob
prob_all<-as.data.frame(final_score)
pred.test_all<-data.frame(pred.test,prob_all)

ggplot(pred.test_all,aes(x=prob))+geom_density()+xlim(0.2,1)

#write.csv(pred.test_all,"final_data.csv")
pred.test_all<-pred.test_all[order(-pred.test_all$prob),]
pred.test_all<-read.csv("final_data.csv")
#pred.test_all<-pred.test_all[order(-pred.test_all$final_MIPS_score.x),]

m<-nrow(pred.test_all)
num<-m
picture <- data.frame(quan=rep(NA,num),score=rep(NA,num),assgn_pen=rep(NA,num)) 

for (i in 1:m){
  set<-pred.test_all[1:i,]
  picture$quan[i]<-i/m
  picture$score[i]<-mean(set$final_MIPS_score.x)
  picture$assgn_pen[i]<-sum(as.numeric(set$assgn)-1)/i
}
summary(picture)


set<-pred.test_all[order(pred.test_all$final_score),]
set_test<-pred.test_all[1:3382,]
summary(set_test)
summary(pred.test_all)

library(plotly) 
#plot_ly(showscale = TRUE) %>% add_surface(z = ~assgn_pen) %>% add_surface(z = ~quan, opacity = 0.98) %>% add_surface(z = ~score, opacity = 0.98) 
plot3d(picture[c('quan','score','assgn_pen')],interactive=TRUE,xlab="quan",ylab="score",zlab="assgn_pen")

plot(picture$quan,picture$score,type="l",main="Relationship between doctor performance and threshold",xlab="Threshold",ylab="Doctor performance")
plot(picture$quan,picture$assgn_pen,type="l",main="Relationship between percentage of accepting doctor and threshold",xlab="Threshold",ylab="Percentage of accepting doctor")
