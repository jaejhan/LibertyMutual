# Kaggle competition: Libery Mutual (July-August 2015)
# Set Global environment ----
setwd('/Users/jaehan/Dropbox/Kaggle/Liberty')
library(plyr); library(reshape2); library(ggplot2); library(lubridate); library(stringr); library(dplyr); library(scales); library(tidyr); library(ggvis)
library(Metrics); library(gbm); library(randomForest); library(glmnet); library(pROC); library(extraTrees); library(xgboost)
library(foreach); library(doParallel); library(beepr); library(rpivotTable); library(minerva); library(corrplot); library(readr); library(Ckmeans.1d.dp)
v <- View; s <- subset; n <- names; h <- head; u <- unique; co <- count; wc <- write.csv

source('src/02_Liberty.R'); source('src/99_Model_utilities.R')

# Load datasets ----
train <- read.csv('data/train.csv', stringsAsFactors=TRUE)
test <- read.csv('data/test.csv', stringsAsFactors=TRUE)

# Add features ----
train <- makeNumeric(train, c('T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14', 'T2_V1', 'T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14', 'T2_V15'))

# Shuffle data & add CV folds
set.seed(2015); train <- train[sample(nrow(train)), ]  # Shuffle data in case ordering leads to biased cross-validation
train$rowid <- 1:nrow(train)
train$cvFold <- ntile(train$rowid, 2)

# Get summary statistics & visualizations of variables ----
# summStats_train <- getSummaryStats(train, names(train), yvar='Hazard', export=TRUE); summStats_test <- getSummaryStats(test, names(test), yvar='none', export=TRUE)
# freq_train <- getFactorFreqs(train, names(train), yvar='Hazard', export=TRUE); freq_test <- getFactorFreqs(test, names(test), yvar='none', export=TRUE) 
# hist(train$Hazard)
# rpivotTable(train)
# ggvis(train, x=~Hazard)
# 
# M <- cor(train[sapply(train, function(x) !is.factor(x))])
# corrplot(M, method = "number",order = "hclust",type='lower', diag=F, addCoefasPercent=T) 
# M <- cor(train_numeric[sapply(train_numeric, function(x) !is.character(x))])
# corrplot.mixed(M, order = "alphabet", lower = "circle", upper = "number", tl.cex = 0.8)

############################################################################################################
# Perform variable selection
e00 <- as.formula('Hazard ~ T1_V1 + T1_V2 + T1_V3 + T1_V4 + T1_V5 + T1_V6 + T1_V7 + T1_V8 + T1_V9 + T1_V10 + T1_V11 + T1_V12 + T1_V13 + T1_V14 + T1_V15 + T1_V16 + T1_V17 + T2_V1 + T2_V2 + T2_V3 + T2_V4 + T2_V5 + T2_V6 + T2_V7 + T2_V8 + T2_V9 + T2_V10 + T2_V11 + T2_V12 + T2_V13 + T2_V14 + T2_V15')
# T2_V7, T2_V10, T1_V10, T1_V13, T2_V12
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
train$logHazard <- log(train$Hazard)
e03b <- as.formula('logHazard ~ T1_V8 + T1_V1 + T1_V4 + T1_V2 + T1_V11_HazardRate + T2_V1  + T1_V5_HazardRate + 
                  T2_V2 + T1_V12_HazardRate + T2_V9 + T1_V16_HazardRate + T2_V15 + T2_V4 +  
                  T2_V13 + T1_V3 + T1_V15_HazardRate + T1_V7_HazardRate + T2_V14 + T1_V14 +  
                  T2_V6 + T2_V8 + T2_V5 + T1_V9 + T2_V11 + T1_V17 + T2_V3 + T1_V6 + T2_V12')


e04 <- as.formula('Hazard ~ T1_V8 + T1_V1 + T1_V4 + T1_V2 + T1_V11 + T2_V1  + T1_V5 + 
                  T2_V2 + T1_V12 + T2_V9 + T1_V16 + T2_V15 + T2_V4 +  
                  T2_V13 + T1_V3 + T1_V15 + T1_V7 + T2_V14 + T1_V14 +  
                  T2_V6 + T2_V8 + T2_V5 + T1_V9 + T2_V11 + T1_V17 + T2_V3 + T1_V6 + T2_V12')

e05 <- as.formula('log(Hazard) ~ T1_V8 + T1_V1 + T1_V4 + T1_V2 + T1_V11_HazardRate + T2_V1  + T1_V5 + 
                  T2_V2 + T1_V12 + T2_V9 + T1_V16_HazardRate + T2_V15 + T2_V4 +  
                  T2_V13 + T1_V3 + T1_V15_HazardRate  + T1_V7_HazardRate + T2_V14 + T1_V14 +  
                  T2_V6 + T2_V8 + T2_V5 + T1_V9 + T2_V11 + T1_V17 + T2_V3 + T1_V6 + T2_V12')


############################################################################################################
# Tune models
# Tune XGB


system.time({
  xgbGrid <- expand.grid(nround = c(650), 
                         subsample = c(0.65), # 0.7
                         eta = c(0.01),  # 0.005
                         max.depth = c(9),  # 9
                         min.child.weight = c(75),  # 6
                         colsample_bytree = c(0.65) # 0.7
  )
  
#tune_results_xgb <- runXgbTuning(xgbGrid, train, model=e03, obj='reg:linear', cv_fold='cvFold', alert=TRUE) 
#tune_results_xgb <- runXgbTuning_par1(xgbGrid, train, model=e03, obj='reg:linear', cv_fold='cvFold', cores=6, alert=TRUE) 
tune_results_xgb <- runXgbTuning_par2(xgbGrid, train, model=e03b, obj='count:poisson', numSteps=10, stepSize=50, cv_fold='cvFold', cores=7, alert=TRUE) 
})

tune_results_xgb_2=filter(tune_results_xgb, nround>=550, min.child.weight==75)
varImp_xgb_2= rename(varImp_xgb, importance=avg_importance)
getXgbTuningParamPlots(tune_results_xgb_2, varImp_xgb_2)

# Tune GBM

system.time({

gbmGrid <- expand.grid(bag.fraction = c(0.7),
                       shrinkage = c(0.01), 
                       interaction.depth = c(13), 
                       n.minobsinnode = c(100),
                       trees = c(2300)
                       )
  
tune_results_gbm <- runGbmTuning_par1(gbmGrid, train, model=e03, var.monotone=constraints03, distrib='poisson', cv_fold='cvFold', alert=T, cores=4) 
#tune_results_gbm <- runGbmTuning_par2(gbmGrid, train, model=e06, var.monotone=constraints04, distrib='poisson', cv_fold='cvFold', numSteps=0, stepSize=0, alert=T, cores=2) 
})
test <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=200)

# tune_results_gbm <- runGbmTuning_par2(gbmGrid, train, model=e02, var.monotone=constraints02, distrib='poisson', numSteps=0, stepSize=100, cv_fold='cvFold', alert=T, cores=4) 
getPDplots_scaled(gbmModel, varList=c('T1_V5_HazardRate'), n.trees=2300, validation_gbm)
getPDplots_scaled(gbmModel, varList=c('T2_V12'), n.trees=2300, train)

tune_results_gbm_2=subset(tune_results_gbm, trees <= 2600 & trees >=2200)
tune_results_gbm_2=subset(tune_results_gbm_2, interaction.depth==13)
tune_relinf_2= rename(tune_relinf, rel.inf=avg_rel.inf)
getGbmTuningParamPlots2(tune_results_gbm_2, tune_relinf_2)

write.csv(validation_collect, file='data/validation_collect.csv');

# Tune Random forests
system.time({

rfGrid <- expand.grid(trees = c(1000), 
                      mtry = c(6), 
                      maxnodes = c(50, 75, 100),
                      nodesize = c(300, 500, 700),
                      sampsize = c(15000),
                      replace = c(TRUE)
                      )
#tune_results_rfo <- runRfTuning_par1(rfGrid, train, model=e05, cv_fold='cvFold', alert=T, cores=6)                                                     
tune_results_rfo <- runRfTuning_par1(rfGrid, train, model=e05, cv_fold='cvFold', alert=T, cores=4)                                                   

})

tune_results_rfo_2=subset(tune_results_rfo, nodesize == 300)
tune_results_rfo_2=subset(tune_results_rfo_2, sampsize==15000)
varImp_rfo_2= rename(varImp_rfo, importance=avg_importance)
getRfTuningParamPlots(tune_results_rfo_2, varImp_rfo_2)

# Tune GLMNET
gntGrid <- expand.grid(alpha = c(0.5), 
                       lambda.min.ratio = c(0.0001))

tune_results_gnt <- runGntTuning(gntGrid, train, model=e20, family='binomial', cv_fold='cvFold') 


# Ensemble (CV score)
validation_rfo <- validation_rfo[, c('enrollment_id', 'score_rfo')]
validation_xgb <- validation_xgb[, c('enrollment_id', 'score_xgb')]
#validation_gnt <- validation_gnt[, c('enrollment_id', 'score_gnt')]

train_ensemble <- left_join(validation_gbm, validation_rfo, by='enrollment_id')
#train_ensemble <- left_join(train_ensemble, validation_gnt, by='enrollment_id')
train_ensemble <- left_join(train_ensemble, validation_xgb, by='enrollment_id')

train_ensemble$score_ensemble <- (0.65 * train_ensemble$score_gbm) + 
                                 (0.1 * train_ensemble$score_rfo) +
                                 (0.25 * train_ensemble$score_xgb)
                                 #(0.05 * train_ensemble$score_gnt)
  
train_ensemble01 <- subset(train_ensemble, cvFold == 1)
train_ensemble02 <- subset(train_ensemble, cvFold == 2)

auc_train1 <- Metrics::auc(train_ensemble01$dropped, train_ensemble01$score_ensemble) 
auc_train2 <- Metrics::auc(train_ensemble02$dropped, train_ensemble02$score_ensemble) 
mean(c(auc_train1, auc_train2))

# auc_summary1 <- train_ensemble01 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_ensemble)) %>% ungroup()
# auc_summary2 <- train_ensemble02 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_ensemble)) %>% ungroup()
# auc_summary <- rbind(auc_summary1, auc_summary2)
# write.csv(auc_summary, file='data/aucauc_summary.csv')
# a = subset(train, uniqueLoginDays == 1 & dropped == 0)
# aa=count(a, course_id)
# bb=count(train, course_id)
# write.csv(aa, file='data/aa.csv')
# write.csv(bb, file='data/bb.csv')
validation_rfo <- validation_rfo[, c('enrollment_id', 'score_rfo')]
validation_xgb <- validation_xgb[, c('enrollment_id', 'score_xgb')]
validation_gnt <- validation_gnt[, c('enrollment_id', 'score_gnt')]

train_ensemble <- left_join(validation_gbm, validation_rfo, by='enrollment_id')
train_ensemble <- left_join(train_ensemble, validation_xgb, by='enrollment_id')
train_ensemble <- left_join(train_ensemble, validation_gnt, by='enrollment_id')

train_ensemble$score_ensemble <- (0.65 * train_ensemble$score_gbm) + 
  (0.09 * train_ensemble$score_rfo) +
  (0.22 * train_ensemble$score_xgb) +
  (0.05 * train_ensemble$score_gnt)

train_ensemble01 <- subset(train_ensemble, cvFold == 1)
train_ensemble02 <- subset(train_ensemble, cvFold == 2)

auc_summary_gbm <- train_ensemble01 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_gbm)) %>% ungroup()
auc_summary_rfo <- train_ensemble01 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_rfo)) %>% ungroup()
auc_summary_xgb <- train_ensemble01 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_xgb)) %>% ungroup()
auc_summary_gnt <- train_ensemble01 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_gnt)) %>% ungroup()

auc_summary_gbm2 <- train_ensemble02 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_gbm)) %>% ungroup()
auc_summary_rfo2 <- train_ensemble02 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_rfo)) %>% ungroup()
auc_summary_xgb2 <- train_ensemble02 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_xgb)) %>% ungroup()
auc_summary_gnt2 <- train_ensemble02 %>% group_by(course_id) %>% summarise(AUC=Metrics::auc(dropped, score_gnt)) %>% ungroup()

write.csv(auc_summary_gbm, file='data/auc_summary_gbm.csv');
write.csv(auc_summary_rfo, file='data/auc_summary_rfo.csv');
write.csv(auc_summary_xgb, file='data/auc_summary_xgb.csv');
write.csv(auc_summary_gnt, file='data/auc_summary_gnt.csv');

write.csv(auc_summary_gbm2, file='data/auc_summary_gbm2.csv');
write.csv(auc_summary_rfo2, file='data/auc_summary_rfo2.csv');
write.csv(auc_summary_xgb2, file='data/auc_summary_xgb2.csv');
write.csv(auc_summary_gnt2, file='data/auc_summary_gnt2.csv');

############################################################################################################
# Run models
############################################################################################################
train <- getOneWayVars(train, test, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0.2)
train <- getOneWayVars(train, test, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
train <- getOneWayVars(train, test, c('T1_V12'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
train <- getOneWayVars(train, test, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
train <- getOneWayVars(train, test, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
train <- getOneWayVars(train, test, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0.2)


xgbGrid <- expand.grid(nround = c(650), 
                       subsample = c(0.65), # 0.7
                       eta = c(0.01),  # 0.005
                       max.depth = c(9),  # 9
                       min.child.weight = c(75),  # 6
                       colsample_bytree = c(0.65) # 0.7
)


xgbModel <- runXGB(train, test, model=e03, watch=T, obj='reg:linear', nround=10000, subsample=0.7, eta=0.005, max_depth=9, min_child_weight=6, colsample_bytree=0.7, offset=5000, feval=evalgini, maximize=TRUE, early.stop.round=500, nthread=2);   

test$score_xgb1 <- predict(xgbModel, xgb.DMatrix(data=sparse.model.matrix(~., data = test[, all.vars(e03)[-1]])))

rfoModel <- runRFO(train, test, model=e05, trees=1500, mtry=6, maxnodes=50, nodesize=300, sampsize=15000, replace=TRUE); test$score_rfo <- predict(rfoModel, test, type='response')
  rfoModel_log <- rfoModel; save(rfoModel_log, file='data/rfoModel_log.RData')

gbmModel <- runGBM(train, test, model=e03, var.monotone=constraints03, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=200); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
  gbmModel_poisson <- gbmModel; save(gbmModel_poisson, file='data/gbmModel_poisson.RData')
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=240); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=120); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=200); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=2300, shrnkg=0.01, idepth=13, minobs=160); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e02, var.monotone=constraints02, distrib='poisson', bag=0.7, trees=450, shrnkg=0.05, idepth=7, minobs=70); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e01, var.monotone=constraints01, distrib='poisson', bag=0.7, trees=450, shrnkg=0.01, idepth=7, minobs=70); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)
gbmModel <- runGBM(train, test, model=e01, var.monotone=constraints01, distrib='poisson', bag=0.7, trees=2500, shrnkg=0.01, idepth=7, minobs=80); test$score_gbm <- predict.gbm(gbmModel, test, type='response', n.trees=gbmModel$n.trees)





test$score_ensemble <- (0.75 * test$score_gbm) + 
                       (0.1 * test$score_rfo) +
                       (0.15 * test$score_xgb) 
#                        (0.25 * test$score_gnt)


############################################################################################################
# Export submission file
exportSubmission(test, '14', cols=c('Id', 'score_xgb1'), replace=c('Id', 'Hazard'))
