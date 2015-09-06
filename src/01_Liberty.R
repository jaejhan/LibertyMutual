# Kaggle competition: Libery Mutual (July-August 2015)
# Set Global environment ----
setwd('/Users/jaehan/Dropbox/Kaggle/Liberty')
library(plyr); library(reshape2); library(ggplot2); library(lubridate); library(stringr); library(dplyr); library(scales); library(tidyr); library(ggvis)
library(Metrics); library(gbm); library(randomForest); library(glmnet); library(pROC); library(extraTrees); library(xgboost)
library(foreach); library(doParallel); library(beepr); library(rpivotTable); library(minerva); library(corrplot); library(readr); library(Ckmeans.1d.dp)
v <- View; s <- subset; n <- names; h <- head; u <- unique; co <- count; wc <- write.csv

source('src/02_Liberty.R'); source('src/99_Model_utilities.R')

# Load datasets ---------------------------------------------------------------------------
train <- read.csv('data/train.csv', stringsAsFactors=TRUE)
test <- read.csv('data/test.csv', stringsAsFactors=TRUE)

# Add features ---------------------------------------------------------------------------
train <- makeNumeric(train, c('T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14', 'T2_V1', 'T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14', 'T2_V15'))
train$T1_V4N_ind <- ifelse(train$T1_V4 == 'N', 1, 0)
test$T1_V4N_ind <- ifelse(test$T1_V4 == 'N', 1, 0)

# Shuffle data & add CV folds ---------------------------------------------------------------------------
set.seed(2015); train <- train[sample(nrow(train)), ]  # Shuffle data in case original order leads to biased cross-validation
train$rowid <- 1:nrow(train)
train$cvFold <- ntile(train$rowid, 5)

# Get summary statistics & visualizations of variables ----
summStats_train <- getSummaryStats(train, names(train), yvar='Hazard', export=TRUE); summStats_test <- getSummaryStats(test, names(test), yvar='none', export=TRUE)
freq_train <- getFactorFreqs(train, names(train), yvar='Hazard', export=TRUE); freq_test <- getFactorFreqs(test, names(test), yvar='none', export=TRUE) 
hist(train$Hazard)
rpivotTable(train)
ggvis(train, x=~Hazard)
 
M <- cor(train[sapply(train, function(x) !is.factor(x))])
corrplot(M, method = "number",order = "hclust",type='lower', diag=F, addCoefasPercent=T) 
M <- cor(train_numeric[sapply(train_numeric, function(x) !is.character(x))])
corrplot.mixed(M, order = "alphabet", lower = "circle", upper = "number", tl.cex = 0.8)


# Model formulas ----------------------------------------------------------------------------
e00 <- as.formula('Hazard ~ T1_V1 + T1_V2 + T1_V3 + T1_V4 + T1_V5 + T1_V6 + T1_V7 + T1_V8 + T1_V9 + T1_V10 + T1_V11 + T1_V12 + T1_V13 + T1_V14 + T1_V15 + T1_V16 + T1_V17 + T2_V1 + T2_V2 + T2_V3 + T2_V4 + T2_V5 + T2_V6 + T2_V7 + T2_V8 + T2_V9 + T2_V10 + T2_V11 + T2_V12 + T2_V13 + T2_V14 + T2_V15')

# T2_V7, T2_V10, T1_V10, T1_V13 removed
e01 <- as.formula('Hazard ~ T1_V1 + T1_V2 + T2_V1 + T2_V2 + T2_V15 + T2_V9 + T2_V4 + T1_V3 + 
                  T2_V14 + T1_V14 + T2_V6 + T2_V8 + T1_V8_HazardRate + T1_V16_HazardRate + 
                  T1_V11_HazardRate + T1_V12_HazardRate + T1_V4 + T1_V5 + T1_V15_HazardRate +
                  T1_V7_HazardRate + T2_V13 + T2_V5 + T1_V9 + T2_V11 + T1_V17 + T2_V3 +
                  T1_V6')
constraints01 <- c(0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0)

e02 <- as.formula('Hazard ~ T1_V8 + T1_V1 + T1_V4 + T1_V2 + T1_V11_HazardRate + T2_V1  + T1_V5_HazardRate + 
                  T2_V2 + T1_V12_HazardRate + T2_V9 + T1_V16_HazardRate + T2_V15 + T2_V4 +  
                  T2_V13 + T1_V3 + T1_V15_HazardRate + T1_V7_HazardRate + T2_V14 + T1_V14 +  
                  T2_V6 + T2_V8 + T2_V5 + T1_V9 + T2_V11 + T1_V17 + T2_V3 + T1_V6')
constraints02 <- c(0, 0, 0, 0, 1, -1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

e03 <- as.formula('Hazard ~ T1_V8 + T1_V1 + T1_V4 + T1_V2 + T1_V11_HazardRate + T2_V1  + T1_V5_HazardRate + 
                  T2_V2 + T1_V12_HazardRate + T2_V9 + T1_V16_HazardRate + T2_V15 + T2_V4 +  
                  T2_V13 + T1_V3 + T1_V15_HazardRate + T1_V7_HazardRate + T2_V14 + T1_V14 +  
                  T2_V6 + T2_V8 + T2_V5 + T1_V9 + T2_V11 + T1_V17 + T2_V3 + T1_V6 + T2_V12')
constraints03 <- c(0, 0, 0, 0, 1, -1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

e04 <- as.formula('Hazard ~ T1_V8 + T1_V1 + T1_V4 + T1_V2 + T1_V11 + T2_V1  + T1_V5 + 
                  T2_V2 + T1_V12 + T2_V9 + T1_V16 + T2_V15 + T2_V4 +  
                  T2_V13 + T1_V3 + T1_V15 + T1_V7 + T2_V14 + T1_V14 +  
                  T2_V6 + T2_V8 + T2_V5 + T1_V9 + T2_V11 + T1_V17 + T2_V3 + T1_V6 + T2_V12')

e05 <- as.formula('log(Hazard) ~ T1_V8 + T1_V1 + T1_V4 + T1_V2 + T1_V11_HazardRate + T2_V1  + T1_V5 + 
                  T2_V2 + T1_V12 + T2_V9 + T1_V16_HazardRate + T2_V15 + T2_V4 +  
                  T2_V13 + T1_V3 + T1_V15_HazardRate  + T1_V7_HazardRate + T2_V14 + T1_V14 +  
                  T2_V6 + T2_V8 + T2_V5 + T1_V9 + T2_V11 + T1_V17 + T2_V3 + T1_V6 + T2_V12')

e06 <- as.formula('Hazard ~ T1_V8 + T1_V1 + T1_V4 + T1_V2 + T1_V11 + T2_V1  + T1_V5 + 
                   T2_V2 + T1_V12 + T2_V9 + T1_V16 + T2_V15 + T2_V4 +  
                   T2_V13 + T1_V3 + T1_V15 + T1_V7 + T2_V14 + T1_V14 +  
                   T2_V6 + T2_V8 + T2_V5 + T1_V9 + T2_V11 + T1_V17 + T2_V3 + T1_V6 + T2_V12')

e07 <- as.formula('Hazard ~ T1_V8_HazardRate + T1_V1 + T1_V4_HazardRate + T1_V2 + T1_V11_HazardRate + T2_V1  + T1_V5_HazardRate + 
                   T2_V2 + T1_V12_HazardRate + T2_V9 + T1_V16_HazardRate + T2_V15 + T2_V4 +  
                  T2_V13_HazardRate + T1_V3 + T1_V15_HazardRate + T1_V7_HazardRate + T2_V14 + T1_V14 +  
                  T2_V6 + T2_V8 + T2_V5_HazardRate + T1_V9_HazardRate + T2_V11_HazardRate + T1_V17_HazardRate + T2_V3_HazardRate + T1_V6_HazardRate + T2_V12_HazardRate')

e08 <- as.formula('Hazard ~ T1_V8_HazardRate + T1_V4_HazardRate + T1_V11_HazardRate + T1_V5_HazardRate + 
                   T1_V12_HazardRate + T1_V16_HazardRate + T2_V13_HazardRate + T1_V15_HazardRate + T1_V7_HazardRate +  
                  T2_V5_HazardRate + T1_V9_HazardRate + T2_V11_HazardRate + T1_V17_HazardRate + T2_V3_HazardRate + T1_V6_HazardRate + T2_V12_HazardRate')


# Tune models -------------------------------------------------------------------------------
# Tune XGB
train <- left_join(train, validation_rfo, by='Id')

xgbGrid <- expand.grid(nround = c(650), 
                       subsample = c(0.65),
                       eta = c(0.01),  
                       max.depth = c(9),  
                       min.child.weight = c(75),  
                       colsample_bytree = c(0.65))
                      
#tune_results_xgb <- runXgbTuning(xgbGrid, train, model=e03, obj='reg:linear', cv_fold='cvFold', alert=TRUE) 
#tune_results_xgb <- runXgbTuning_par1(xgbGrid, train, model=e03, obj='reg:linear', cv_fold='cvFold', cores=6, alert=TRUE) 
tune_results_xgb <- runXgbTuning_par2(xgbGrid, train, model=e03, obj='reg:linear', numSteps=10, stepSize=50, cv_fold='cvFold', cores=2, alert=TRUE) 

tune_results_xgb_2=filter(tune_results_xgb, nround>=550, min.child.weight==75)
varImp_xgb_2= rename(varImp_xgb, importance=avg_importance)
getXgbTuningParamPlots(tune_results_xgb_2, varImp_xgb_2)

# Tune GBM
gbmGrid <- expand.grid(bag.fraction = c(0.7),
                       shrinkage = c(0.01), 
                       interaction.depth = c(14, 15, 16, 17), 
                       n.minobsinnode = c(100, 120, 140, 160, 180),
                       trees = c(2900))
                       
#tune_results_gbm <- runGbmTuning(gbmGrid, train, model=e03, var.monotone=constraints03, distrib='poisson', cv_fold='cvFold', alert=T) 
#tune_results_gbm <- runGbmTuning_par1(gbmGrid, train, model=e03, var.monotone=constraints03b, distrib='poisson', cv_fold='cvFold', alert=T, cores=4) 
tune_results_gbm <- runGbmTuning_par2(gbmGrid, train, model=e03, var.monotone=constraints03, distrib='poisson', cv_fold='cvFold', numSteps=40, stepSize=50, alert=T, cores=6) 


getPDplots_scaled(gbmModel, varList=c('T2_V12'), n.trees=2300, train)

tune_results_gbm_2=subset(tune_results_gbm, trees >= 2000 & trees <= 2500)
tune_results_gbm_2=subset(tune_results_gbm_2, interaction.depth==15)
varImp_gbm_2= rename(varImp_gbm, rel.inf=avg_rel.inf)
getGbmTuningParamPlots(tune_results_gbm_2, varImp_gbm_2)


# Tune Random forests
rfGrid <- expand.grid(trees = c(1500), 
                      mtry = c(6), 
                      maxnodes = c(50),
                      nodesize = c(300),
                      sampsize = c(15000),
                      replace = c(TRUE))
                      
tune_results_rfo <- runRfTuning(rfGrid, train, model=e05, cv_fold='cvFold', alert=T)                                               
#tune_results_rfo <- runRfTuning_par1(rfGrid, train, model=e05, cv_fold='cvFold', alert=T, cores=4)                                                   

tune_results_rfo_2=subset(tune_results_rfo, nodesize == 300)
tune_results_rfo_2=subset(tune_results_rfo_2, sampsize==15000)
varImp_rfo_2= rename(varImp_rfo, importance=avg_importance)
getRfTuningParamPlots(tune_results_rfo_2, varImp_rfo_2)

# Tune GLMNET
gntGrid <- expand.grid(alpha = c(0), 
                       lambda.min.ratio = c(0.0001))

tune_results_gnt <- runGntTuning_par(gntGrid, train, model=e08, family='gaussian', cv_fold='cvFold') 

# Save
save(validation_gbm, file='data/validation_gbm.RData')
save(validation_rfo, file='data/validation_rfo.RData')
save(validation_xgb, file='data/validation_xgb.RData')
save(validation_gnt, file='data/validation_gnt.RData')

load(file='data/validation_gbm.RData')
load(file='data/validation_rfo.RData')
load(file='data/validation_xgb.RData')
load(file='data/validation_gnt.RData')

# Ensemble (CV score)
validation_rfo <- validation_rfo[, c('Id', 'score_rfo')]
validation_xgb <- validation_xgb[, c('Id', 'score_xgb')]
validation_gnt <- validation_gnt[, c('Id', 'score_gnt')]

train_ensemble <- left_join(validation_gbm, validation_rfo, by='Id')
train_ensemble <- left_join(train_ensemble, validation_xgb, by='Id')
train_ensemble <- left_join(train_ensemble, validation_gnt, by='Id')

train_ensemble$score_ensemble <- (0.4 * train_ensemble$score_gbm) + 
                                 (0.30 * train_ensemble$score_rfo) +
                                 (0.3 * train_ensemble$score_xgb) +

  
train_ensemble01 <- subset(train_ensemble, cvFold == 1)
train_ensemble02 <- subset(train_ensemble, cvFold == 2)
ngini_train1 <- ngini(train_ensemble01$Hazard, train_ensemble01$score_ensemble) 
ngini_train2 <- ngini(train_ensemble02$Hazard, train_ensemble02$score_ensemble) 
mean(c(ngini_train1, ngini_train2))

ngini(train_ensemble01$Hazard, train_ensemble01$score_gbm) 
ngini(train_ensemble02$Hazard, train_ensemble02$score_gbm) 

load(file='data/validation_rfo.RData')
ngini(validation_rfo$Hazard[validation_rfo$cvFold==1], validation_rfo$score_rfo[validation_rfo$cvFold==1])
ngini(train_ensemble01$Hazard, train_ensemble01$score_rfo) 
ngini(train_ensemble02$Hazard, train_ensemble02$score_rfo) 

ngini(train_ensemble01$Hazard, train_ensemble01$score_xgb) 
ngini(train_ensemble02$Hazard, train_ensemble02$score_xgb) 


# Run models -------------------------------------------------------------------------------
train <- getOneWayVars(train, test, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0)
train <- getOneWayVars(train, test, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0)
train <- getOneWayVars(train, test, c('T1_V12'), 'Hazard', freq=TRUE, cred=20, rand=0)
train <- getOneWayVars(train, test, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0)
train <- getOneWayVars(train, test, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0)
train <- getOneWayVars(train, test, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0)


gbmModel <- runGBM(train, test, model=e03, var.monotone=constraints03, distrib='poisson', bag=0.7, trees=2200, shrnkg=0.01, idepth=15, minobs=200)
  gbmModel_poisson3 <- gbmModel; save(gbmModel_poisson3, file='data/gbmModel_poisson3.RData')

gbmModel <- runGBM(train, test, model=e03, var.monotone=constraints03, distrib='poisson', bag=0.7, trees=1600, shrnkg=0.01, idepth=13, minobs=100)
  gbmModel_poisson2 <- gbmModel; save(gbmModel_poisson2, file='data/gbmModel_poisson2.RData')

xgbModel <- runXGB(train, test, model=e07, watch=T, obj='reg:linear', nround=812, subsample=0.7, eta=0.005, max_depth=9, min_child_weight=6, colsample_bytree=0.7, offset=4000, feval=evalgini, maximize=T, early.stop.round=120, nthread=2);   
  xgbModel_03 <- xgbModel; save(xgbModel_03, file='data/xgbModel_03.RData') 

xgbModel <- runXGB(train, test, model=e06, watch=T, obj='reg:linear', nround=809, subsample=0.7, eta=0.005, max_depth=9, min_child_weight=6, colsample_bytree=0.7, offset=4000, feval=evalgini, maximize=T, early.stop.round=120, nthread=2);   
  xgbModel_02 <- xgbModel; save(xgbModel_02, file='data/xgbModel_02.RData')

xgbModel <- runXGB(train, test, model=e03, watch=F, obj='reg:linear', nround=548, subsample=0.65, eta=0.01, max_depth=9, min_child_weight=75, colsample_bytree=0.65, offset=5000, feval=evalgini, maximize=TRUE, early.stop.round=100, nthread=2);   
  xgbModel_01 <- xgbModel; save(xgbModel_01, file='data/xgbModel_01.RData')

rfoModel <- runRFO(train, test, model=e05, trees=1500, mtry=6, maxnodes=50, nodesize=300, sampsize=15000, replace=TRUE)
  rfoModel_log <- rfoModel; save(rfoModel_log, file='data/rfoModel_log.RData')

gbmModel <- runGBM(train, test, model=e03, var.monotone=constraints03, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=200)
  gbmModel_poisson <- gbmModel; save(gbmModel_poisson, file='data/gbmModel_poisson.RData')
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=240); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=120); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=200); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=160); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=450, shrnkg=0.05, idepth=7, minobs=70); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e01, var.monotone=constraints01, distrib='poisson', bag=0.7, trees=450, shrnkg=0.01, idepth=7, minobs=70); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e01, var.monotone=constraints01, distrib='poisson', bag=0.7, trees=2500, shrnkg=0.01, idepth=7, minobs=80); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)


load(file='data/gbmModel_poisson.RData')
load(file='data/rfoModel_log.RData')
load(file='data/xgbModel_01.RData')

test$score_gbm <- predict.gbm(gbmModel_poisson3, test, type='response', n.trees=gbmModel_poisson3$n.trees)

test$score_gbm <- predict.gbm(gbmModel_poisson2, test, type='response', n.trees=gbmModel_poisson2$n.trees)

test$score_xgb2 <- predict(xgbModel_02, xgb.DMatrix(data=sparse.model.matrix(~., data = test[, all.vars(e06)[-1]])))
test$score_xgb3 <- predict(xgbModel_03, xgb.DMatrix(data=sparse.model.matrix(~., data = test[, all.vars(e07)[-1]])))
test$score_xgb_final <- (0.47 * test$score_xgb2^0.045) + (0.53 * test$score_xgb3^0.055)

test$score_gbm <- predict.gbm(gbmModel_poisson, test, type='response', n.trees=gbmModel_poisson$n.trees)
test$score_rfo <- predict(rfoModel_log, test, type='response')
test$score_xgb <- predict(xgbModel_01, xgb.DMatrix(data=sparse.model.matrix(~., data = test[, all.vars(e03)[-1]])))


test$score_ensemble <- (0.4 * test$score_gbm) + 
                       (0.3 * test$score_rfo) +
                       (0.3 * test$score_xgb) 


# Export submission file --------------------------------------------------------------------
exportSubmission(test, '19', cols=c('Id', 'score_gbm'), replace=c('Id', 'Hazard'))
