############################################################################################################
# Function to run extreme GBM model
runXGB <- function(train, test, model, watch, obj, nround, subsample, eta, max_depth, min_child_weight, colsample_bytree, offset, feval, maximize, early.stop.round, nthread=6, alert=TRUE) {
  
  train_predictors <- train[, all.vars(model)[-1]]  # select predictor var columns
  train_predictors <- sparse.model.matrix(~., data = train_predictors)  
  train_y <- as.matrix(train[, all.vars(model)[1]])
  
  if (watch) {
    param <- list("objective" = obj,
                  "subsample" = subsample,
                  "eta" = eta,
                  "max_depth" = max_depth,
                  "min_child_weight" = min_child_weight,
                  "colsample_bytree" = colsample_bytree,
                  "scale_pos_weight" = 1.0)
    
    xgtrain <- xgb.DMatrix(data=train_predictors[offset:nrow(train_predictors), ], label=train_y[offset:nrow(train_predictors)])
    xgval <- xgb.DMatrix(data=train_predictors[1:offset, ], label=train_y[1:offset])
    
    # Setup watchlist (validation must be first for early stopping)
    watchlist <- list(val=xgval, train=xgtrain)
    
    set.seed(1)
    xgbModel <- xgb.train(data = xgtrain, 
                          params = param, 
                          nround = nround, 
                          feval = feval, 
                          maximize = maximize,
                          watchlist = watchlist, 
                          early.stop.round = early.stop.round, 
                          print.every.n = 50)
  }  
  else {
    xgtrain <- xgb.DMatrix(data=train_predictors, label=train_y)
    
    set.seed(1)
    xgbModel <- xgboost(data = xgtrain,
                        objective = obj,  # reg:linear (linear regression); binary:logistic (logistic regression for classification); count:poisson
                        nround = nround,  # maximum number of iterations
                        subsample = subsample, # subsample ratio of the training instance. 
                        eta = eta,  # step size of each boosting step
                        max.depth = max_depth,  # maximum depth of the tree
                        min.child.weight = min_child_weight,  # minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.
                        colsample_bytree = colsample_bytree,  # subsample ratio of columns when constructing each tree
                        print.every.n = 50)
    
  }

  plotFeatureImportance(xgbModel, method='xgb', sparse_matrix=train_predictors)
  
  # Score & calculate evaluate metric for in-sample
  train$score_xgb <- predict(xgbModel, xgb.DMatrix(data=train_predictors))
  ngini_train <- ngini(train[, all.vars(model)[1]], train$score_xgb) 
  cat('\nTrain ngini = ', ngini_train, '.', sep='')

  if (alert) beep(10)
  return(xgbModel)
}


############################################################################################################
# Function to run GBM model
runGBM <- function(train, test, model, var.monotone, distrib, bag, trees, shrnkg, idepth, minobs, alert=TRUE) {
  
  set.seed(1)
    
  gbmModel <- gbm(formula = model,
                  data = train,
                  distribution = distrib, # sets regression loss function
                  bag.fraction = bag,  # introduces robustness and speed
                  n.trees = trees, 
                  shrinkage = shrnkg, # aka learn rate; reduces overfitting
                  interaction.depth = idepth,  # sets the interactions limit
                  n.minobsinnode = minobs,  # manages smoothness
                  var.monotone = var.monotone,
                  #weights = 
                  verbose = FALSE,
                  n.cores = 3)
    
   print(summary(gbmModel, plotit=FALSE))  
#   #gbm.perf(gbmModel, plot.it=TRUE, oobag.curve=FALSE, overlay=FALSE, method='OOB')
   plotFeatureImportance(gbmModel, method='gbm')
    
  # Score & calculate evaluate metric for in-sample
  train$score_gbm <- predict.gbm(gbmModel, train, type='response', n.trees=trees)
  ngini_train <- ngini(train[, all.vars(model)[1]], train$score_gbm) 
  cat('\nTrain ngini = ', ngini_train, '.', sep='')
  
  if (alert) beep(10)
  return(gbmModel)
}

# Function to run Random forest model
runRFO <- function(train, test, model, trees, mtry, maxnodes, nodesize, sampsize, replace, alert=TRUE) {

  set.seed(2)
    
  rfoModel <- randomForest(formula = model,
                           data = train,
                           ntree = trees, # Should not be set to too small to ensure that every input row gets predicted at least a few times
                           mtry = mtry,  # Number of variables randomly sampled as candidates at each split
                           maxnodes = maxnodes, # Maximum number of terminal nodes trees can have. If not given, trees are grown to the maximum possible (subject to limits by nodesize)
                           nodesize = nodesize, # Minimum size of terminal nodes. Setting this number larger causes smaller trees
                           # to be grown (and thus take less time).Default=1 (for classification); Default=5 (for regression)
                           sampsize = sampsize,  # Size of sample to draw
                           replace = replace,
                           do.trace = TRUE,
                           importance = TRUE  # Assess importance of predictors?
                           ) 
                           #proximity = TRUE)
  print(rfoModel)
  #print(importance(rfoModel)); 
  varImpPlot(rfoModel)
  #plotFeatureImportance(rfoModel, method='rfo')
    
  # Score & calculate evaluate metric for in-sample
  train$score_rfo <- predict(rfoModel, train, type='response')
  ngini_train <- ngini(train[, all.vars(model)[1]], train$score_rfo) 
  cat('\nTrain ngini = ', ngini_train, '.', sep='')
  
  if (alert) beep(10)
  return(rfoModel)
}


############################################################################################################
# Function to run Extra Trees model
runEXT <- function(train, test, num, model, trees, mtry, nodesize, numRandomCuts, evenCuts, alert=TRUE) {
  
  set.seed(2)
  
  extModel <- extraTrees(x = train[, all.vars(model)[-1]],
                         y = train[, all.vars(model)[1]],
                         ntree = trees,
                         mtry = mtry,  # the number of features tried at each node
                         nodesize = nodesize,  # the size of leaves of the tree (default is 5 for regression and 1 for classification)
                         numRandomCuts = numRandomCuts,  # the number of random cuts for each (randomly chosen) feature (default 1, which corresponds to the official ExtraTrees method). The higher the number of cuts the higher the chance of a good cut.
                         evenCuts = evenCuts,  # if FALSE then cutting thresholds are uniformly sampled (default). If TRUE then the range is split into even intervals (the number of intervals is numRandomCuts) and a cut is uniformly sampled from each interval.
                         na.action = 'stop', # "zero" will set all NA to zero and "fuse" will build trees by skipping samples when the chosen feature is NA for them
                         numThreads = 2)  # the number of CPU threads to use (default is 1).
  
  print(extModel)
  
  # Score & calculate evaluate metric for in-sample
  train$score_ext <- predict(extModel, train[, all.vars(model)[-1]])
  ngini_train <- ngini(train[, all.vars(model)[1]], train$score_ext) 
  cat('\nTrain ngini = ', ngini_train, '.', sep='')
  
  if (alert) beep(10)
  return(extModel)
}


############################################################################################################
# Function to run GLMNET model
runGNT <- function(train, test, model, family, alpha, lambda.min.ratio, alert=TRUE) {
  
  set.seed(4)
  
  glmnetModel <- cv.glmnet(x = as.matrix(train[, all.vars(model)[-1]]),
                           y = as.matrix(train[, all.vars(model)[1]]),
                           family = family,  # gaussian, binomial, poisson, multinomial, cox, mgaussian
                           alpha = alpha,  # the elasticnet mixing parameter (penalty); alpha=1 is the lasso penalty, and alpha=0 the ridge penalty
                           lambda.min.ratio = lambda.min.ratio)  #Smallest value for lambda, as a fraction of lambda.max, the (data derived) entry
                           #value (i.e. the smallest value for which all coefficients are zero). The default
                           #depends on the sample size nobs relative to the number of variables nvars.
                           #If nobs > nvars, the default is 0.0001, close to zero. If nobs < nvars,
                           #the default is 0.01.
  print(glmnetModel)
  
  # Score & calculate evaluate metric for in-sample
  train$score_gnt <- predict(glmnetModel, as.matrix(train[, all.vars(model)[-1]]), s='lambda.min', type='response')[, 1]
  ngini_train <- ngini(train[, all.vars(model)[1]], train$score_gnt) 
  cat('\nTrain ngini = ', ngini_train, '.', sep='')
  
  if (alert) beep(10)
  return(glmnetModel)
}


# Function to run GLMNET model (only worth it with large datasets)
runGT_parallel <- function(train, test, model, alpha, lambda.min.ratio, alert=TRUE) {
  registerDoParallel(cores=detectCores(all.tests=TRUE))
  set.seed(4)
  
  glmnetModel <- cv.glmnet(x = as.matrix(train[, all.vars(model)[-1]]),
                           y = as.matrix(train[, all.vars(model)[1]]),
                           family = 'binomial',
                           alpha = alpha,
                           lambda.min.ratio = lambda.min.ratio,
                           parallel=TRUE)
  stopImplicitCluster()
  print(glmnetModel)
  
  # Score & calculate evaluate metric for in-sample
  train$score_gn <- predict(glmnetModel, as.matrix(train[, all.vars(model)[-1]]), s='lambda.min', type='response')[, 1]
  ngini_train <- ngini(train[, all.vars(model)[1]], train$score_gn) 
  cat('\nTrain ngini = ', ngini_train, '.', sep='')
  
  if (alert) beep(10)
  return(glmnetModel)
}


############################################################################################################
# Function to tune xgboost model parameters
runXgbTuning <- function(xgbGrid, data, model, obj, cv_fold, alert=TRUE) {
  
  eval_metric_final <- vector(length=nrow(xgbGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  varImp <- NULL
  
  for (i in 1:nrow(xgbGrid)) {   
    
    cat("\n\n", "########## Start calibration of new model ##########", "\n", sep="");
    print(i)
    cat(' : nround=', xgbGrid$nround[i], 
        ' : subsample=', xgbGrid$subsample[i], 
        ' : eta=', xgbGrid$eta[i], 
        ' : max.depth=', xgbGrid$max.depth[i], 
        ' : min.child.weight=', xgbGrid$min.child.weight[i], 
        ' : colsample_bytree=', xgbGrid$colsample_bytree[i], 
        "\n\n", sep='')
    
    eval_metric_cv <- vector(length=folds)
    validation_collect <- NULL
    
    k <- 1
    
    # Train the model
    for (j in 1:folds) {      
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      # Create xgboost class sparse matrices
      train_predictors <- train[, all.vars(model)[-1]]  # select predictor var columns
      train_predictors <- sparse.model.matrix(~., data = train_predictors)  
      train_y <- as.matrix(train[, all.vars(model)[1]])
      
      xgtrain <- xgb.DMatrix(data=train_predictors, label=train_y)
      
      # Fit model
      set.seed(3)
      
      xgbModel <<- xgboost(data = xgtrain,
                           objective = obj,  # reg:linear (linear regression); binary:logistic (logistic regression for classification); count:poisson
                           nround = xgbGrid$nround[i],  # maximum number of iterations
                           subsample = xgbGrid$subsample[i], # subsample ratio of the training instance. 
                           eta = xgbGrid$eta[i],  # step size of each boosting step
                           max.depth = xgbGrid$max.depth[i],  # maximum depth of the tree
                           min.child.weight = xgbGrid$min.child.weight[i],  # minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.
                           colsample_bytree = xgbGrid$colsample_bytree[i],  # subsample ratio of columns when constructing each tree
                           verbose = 0)  # 0: none; 1: performance info; 2: print all info
      
      featureImportance <- as.data.frame(xgb.importance(train_predictors@Dimnames[[2]], model=xgbModel))
      varImp <- rbind(varImp, data.frame(id=i, var=featureImportance[, 1], importance=featureImportance[, 2]))
      
      # Score
      validation_predictors <- validation[, all.vars(model)[-1]]  # select predictor var columns
      validation_predictors <- sparse.model.matrix(~., data = validation_predictors)  
      validation$score_xgb <- predict(xgbModel, xgb.DMatrix(data=validation_predictors))

      validation_collect <- rbind(validation_collect, validation)
      
      eval_metric_cv[k] <- ngini(validation[, all.vars(model)[1]], validation$score_xgb)
      
      k <- k + 1
      
    }
    eval_metric_final[i] <- mean(eval_metric_cv)
    cat('\nEval. metric: ', eval_metric_final[i], sep='')
    
    tmp_tune_results <<- arrange(cbind(xgbGrid, eval_metric_final), desc(eval_metric_final))  # For early stoppages
  }
  
  tune_results <- arrange(cbind(xgbGrid, eval_metric_final), desc(eval_metric_final))
  
  View(tune_results)
  
  getXgbTuningParamPlots(tune_results, varImp)  # Visualize results
  varImp <<- varImp
  varImp <- varImp %>% group_by(var) %>% summarise(avg_importance=mean(importance)) %>% arrange(desc(avg_importance))
  varImp_xgb <<- varImp
  
  validation_xgb <<- validation_collect
  
  if (alert) beep(10)
  return(tune_results)
}

runXgbTuning_par1 <- function(xgbGrid, data, model, obj, cv_fold, alert=TRUE, cores=4) {
  
  eval_metric_final <- vector(length=nrow(xgbGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  varImp <- NULL
  
  registerDoParallel(cores)
  comb <- function(x, ...) {  
    mapply(rbind,x,...,SIMPLIFY=FALSE)
  }
  
  collector <- foreach (i=1:nrow(xgbGrid), .combine='comb', .multicombine=TRUE) %dopar% {
    eval_metric_cv <- foreach (j=1:folds, .combine='c') %do% {
      
      validation_collect <- NULL
      
      # Train the model
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      # Create xgboost class sparse matrices
      train_predictors <- train[, all.vars(model)[-1]]  # select predictor var columns
      train_predictors <- sparse.model.matrix(~., data = train_predictors)  
      train_y <- as.matrix(train[, all.vars(model)[1]])
      
      xgtrain <- xgb.DMatrix(data=train_predictors, label=train_y)
      
      # Fit model
      set.seed(3)
      
      xgbModel <- xgboost(data = xgtrain,
                           objective = obj,  # reg:linear (linear regression); binary:logistic (logistic regression for classification); count:poisson
                           nround = xgbGrid$nround[i],  # maximum number of iterations
                           subsample = xgbGrid$subsample[i], # subsample ratio of the training instance. 
                           eta = xgbGrid$eta[i],  # step size of each boosting step
                           max.depth = xgbGrid$max.depth[i],  # maximum depth of the tree
                           min.child.weight = xgbGrid$min.child.weight[i],  # minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.
                           colsample_bytree = xgbGrid$colsample_bytree[i],  # subsample ratio of columns when constructing each tree
                           verbose = 0)  # 0: none; 1: performance info; 2: print all info
      
      featureImportance <- as.data.frame(xgb.importance(train_predictors@Dimnames[[2]], model=xgbModel))
      varImp <- rbind(varImp, data.frame(id=i, var=featureImportance[, 1], importance=featureImportance[, 2]))
      
      # Score
      validation_predictors <- validation[, all.vars(model)[-1]]  # select predictor var columns
      validation_predictors <- sparse.model.matrix(~., data = validation_predictors)  
      validation$score_xgb <- predict(xgbModel, xgb.DMatrix(data=validation_predictors))
      
      ngini(validation[, all.vars(model)[1]], validation$score_xgb)
    }    
    
    eval_metric_final[i] <- mean(eval_metric_cv)
    
    list(
      data.frame(nround = xgbGrid$nround[i],  
                 subsample = xgbGrid$subsample[i], 
                 eta = xgbGrid$eta[i],  
                 max.depth = xgbGrid$max.depth[i],
                 min.child.weight = xgbGrid$min.child.weight[i], 
                 colsample_bytree = xgbGrid$colsample_bytree[i], 
                 eval_metric_final=eval_metric_final[i]),
      data.frame(varImp),
      data.frame(validation_collect)
    )
  }
  stopImplicitCluster()
  tune_results <- as.data.frame(collector[1])
  tune_results <- arrange(tune_results, desc(eval_metric_final))
  View(tune_results)
  
  varImp <- as.data.frame(collector[2])
  getXgbTuningParamPlots(tune_results, varImp)  # Visualize results
  
  varImp <- varImp %>% group_by(var) %>% summarise(avg_importance=mean(importance)) %>% arrange(desc(avg_importance))
  varImp_xgb <<- varImp
  
  validation_xgb <<- as.data.frame(collector[3])
  
  if (alert) beep(10)
  return(tune_results)
}

runXgbTuning_par2 <- function(xgbGrid, data, model, obj, cv_fold, numSteps=0, stepSize, alert=TRUE, cores=4) {
  
  eval_metric_final <- vector(length=nrow(xgbGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  varImp <- NULL
  
  registerDoParallel(cores)
  comb <- function(x, ...) {  
    mapply(rbind,x,...,SIMPLIFY=FALSE)
  }
  
  collector <- foreach (i=1:nrow(xgbGrid), .combine='comb', .multicombine=TRUE) %dopar% {
    
    eval_metric_cv <- replicate(numSteps + 1, vector(length=folds), simplify=FALSE)
    
    validation_collect <- NULL
    
    k <- 1
    
    # Train the model
    for (j in 1:folds) {      
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      # Create xgboost class sparse matrices
      train_predictors <- train[, all.vars(model)[-1]]  # select predictor var columns
      train_predictors <- sparse.model.matrix(~., data = train_predictors)  
      train_y <- as.matrix(train[, all.vars(model)[1]])
      
      xgtrain <- xgb.DMatrix(data=train_predictors, label=train_y)
      
      # Fit model
      set.seed(3)
      
      xgbModel <- xgboost(data = xgtrain,
                          objective = obj,  # reg:linear (linear regression); binary:logistic (logistic regression for classification); count:poisson
                          nround = xgbGrid$nround[i],  # maximum number of iterations
                          subsample = xgbGrid$subsample[i], # subsample ratio of the training instance. 
                          eta = xgbGrid$eta[i],  # step size of each boosting step
                          max.depth = xgbGrid$max.depth[i],  # maximum depth of the tree
                          min.child.weight = xgbGrid$min.child.weight[i],  # minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.
                          colsample_bytree = xgbGrid$colsample_bytree[i],  # subsample ratio of columns when constructing each tree
                          verbose = 0)  # 0: none; 1: performance info; 2: print all info
      
      featureImportance <- as.data.frame(xgb.importance(train_predictors@Dimnames[[2]], model=xgbModel))
      varImp <- rbind(varImp, data.frame(id=i, var=featureImportance[, 1], importance=featureImportance[, 2]))
      
      # Score
      for (m in 1:(numSteps + 1)) {
        
        if (m > 1) {
          validation_predictors <- validation[, all.vars(model)[-1]]  # select predictor var columns
          validation_predictors <- sparse.model.matrix(~., data = validation_predictors)  

          eval_metric_cv[[m]][[k]] <- ngini(validation[, all.vars(model)[1]], predict(xgbModel, xgb.DMatrix(data=validation_predictors), ntreelimit=xgbGrid$nround[i] - ((m-1) * stepSize)))
        }
        else {
          validation_predictors <- validation[, all.vars(model)[-1]]  # select predictor var columns
          validation_predictors <- sparse.model.matrix(~., data = validation_predictors)  
          validation$score_xgb <- predict(xgbModel, xgb.DMatrix(data=validation_predictors))
          
          validation_collect <- rbind(validation_collect, validation)
          
          eval_metric_cv[[1]][[k]] <- ngini(validation[, all.vars(model)[1]], validation$score_xgb)
          
        }
        
      }
      
      
      k <- k + 1
    }    
    
    xgbGrid_collect <- NULL
    
    for (m in 1:(numSteps + 1)) {
      
      xgbGrid_tmp <- data.frame(nround = xgbGrid$nround[i] - ((m-1) * stepSize),
                                subsample = xgbGrid$subsample[i], 
                                eta = xgbGrid$eta[i],  
                                max.depth = xgbGrid$max.depth[i],
                                min.child.weight = xgbGrid$min.child.weight[i],
                                colsample_bytree = xgbGrid$colsample_bytree[i],
                                eval_metric_final=mean(eval_metric_cv[[m]]))
      
      xgbGrid_collect <- rbind(xgbGrid_collect, xgbGrid_tmp)
    }
    
    list(
      xgbGrid_collect,
      data.frame(varImp),
      data.frame(validation_collect)
    )
  }
  stopImplicitCluster()
  tune_results <- as.data.frame(collector[1])
  tune_results <- arrange(tune_results, desc(eval_metric_final))
  View(tune_results)
  
  varImp <- as.data.frame(collector[2])
  getXgbTuningParamPlots(tune_results, varImp)  # Visualize results
  
  varImp <- varImp %>% group_by(var) %>% summarise(avg_importance=mean(importance)) %>% arrange(desc(avg_importance))
  varImp_xgb <<- varImp
  
  validation_xgb <<- as.data.frame(collector[3])
  
  if (alert) beep(10)
  return(tune_results)
}



############################################################################################################
# Function to tune GBM model parameters (manually do cross-validation using specified evaluation metric)
runGbmTuning <- function(gbmGrid, data, model, var.monotone, distrib, cv_fold, alert=TRUE) {
  
  eval_metric_final <- vector(length=nrow(gbmGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  varImp <- NULL
    
  for (i in 1:nrow(gbmGrid)) {   
    
    cat("\n\n", "########## Start calibration of new model ##########", "\n", sep="");
    print(i)
    cat('Bag fraction=', gbmGrid$bag.fraction[i], 
        ' : Shrinkage=', gbmGrid$shrinkage[i], 
        ' : Interaction depth=', gbmGrid$interaction.depth[i], 
        ' : n.minobsinnode=', gbmGrid$n.minobsinnode[i], 
        "\n\n", sep='')
    
    eval_metric_cv <- vector(length=folds)
    validation_collect <- NULL
    
    k <- 1
    
    # Train the model
    for (j in 1:folds) {      
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)

      train <- getOneWayVars_retTrain(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      set.seed(1)
    
      gbmModel <<- gbm(formula = model,
                     data = train,
                     distribution = distrib,
                     bag.fraction = gbmGrid$bag.fraction[i],
                     n.trees = gbmGrid$trees[i], 
                     shrinkage = gbmGrid$shrinkage[i],
                     interaction.depth = gbmGrid$interaction.depth[i],
                     n.minobsinnode = gbmGrid$n.minobsinnode[i],
                     var.monotone = var.monotone,
                     verbose = TRUE,
                     #class.stratify.cv = gbmGrid$class.stratify.cv[i],
                     n.cores = 4)
    
      varImp <- rbind(varImp, cbind(id=i, summary(gbmModel, plotit=FALSE)))
    
      # Score
      validation$score_gbm <- predict.gbm(gbmModel, validation, type='response', n.trees=gbmGrid$trees[i])
      validation_collect <- rbind(validation_collect, validation)
      
      eval_metric_cv[k] <- ngini(validation[, all.vars(model)[1]], validation$score_gbm)
      
      k <- k + 1
    }    
    
    print(summary(gbmModel, plotit=FALSE))
    eval_metric_final[i] <- mean(eval_metric_cv)
    cat('\nEval. metric: ', eval_metric_final[i], sep='')
    
    tmp_tune_results <<- arrange(cbind(gbmGrid, eval_metric_final), desc(eval_metric_final))  # For early stoppages
  }
  
  tune_results <- arrange(cbind(gbmGrid, eval_metric_final), desc(eval_metric_final))
  View(tune_results)
  
  getGbmTuningParamPlots(tune_results, varImp)  # Visualize results
  
  varImp <- varImp %>% group_by(var) %>% summarise(avg_rel.inf=mean(rel.inf)) %>% arrange(desc(avg_rel.inf))
  varImp_gbm <<- varImp
  
  validation_gbm <<- validation_collect
  
  if (alert) beep(10)
  
  return(tune_results)
}

runGbmTuning_par1 <- function(gbmGrid, data, model, distrib, var.monotone, cv_fold, alert=TRUE, cores=4) {
  
  eval_metric_final <- vector(length=nrow(gbmGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  varImp <- NULL
  
  registerDoParallel(cores)
  comb <- function(x, ...) {  
    mapply(rbind,x,...,SIMPLIFY=FALSE)
  }
  
  collector <- foreach (i=1:nrow(gbmGrid), .combine='comb', .multicombine=TRUE) %dopar% {
    eval_metric_cv <- foreach (j=1:folds, .combine='c') %do% {
      
      validation_collect <- NULL
      
      # Train the model
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V4'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V4'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      
      set.seed(1)
      
      gbmModel <<- gbm(formula = model,
                       data = train,
                       #distribution = gbmGrid$distrib[i],
                       distribution = distrib,
                       bag.fraction = gbmGrid$bag.fraction[i],
                       n.trees = gbmGrid$trees[i], 
                       shrinkage = gbmGrid$shrinkage[i],
                       interaction.depth = gbmGrid$interaction.depth[i],
                       n.minobsinnode = gbmGrid$n.minobsinnode[i],
                       var.monotone = var.monotone,
                       #verbose = TRUE, # won't print anyway with parallelization
                       n.cores = 1)
      
      varImp <- rbind(varImp, cbind(id=i, summary(gbmModel, plotit=FALSE)))
      
      # Score
      validation$score_gbm <- predict.gbm(gbmModel, validation, type='response', n.trees=gbmGrid$trees[i])
      validation_collect <- rbind(validation_collect, validation)
      
      ngini(validation[, all.vars(model)[1]], validation$score_gbm)
    }    
    
    eval_metric_final[i] <- mean(eval_metric_cv)
    
    list(
      data.frame(
        #distrib = gbmGrid$distrib[i],  
        bag.fraction = gbmGrid$bag.fraction[i],  
        shrinkage = gbmGrid$shrinkage[i], 
        interaction.depth = gbmGrid$interaction.depth[i],  
        n.minobsinnode = gbmGrid$n.minobsinnode[i],  
        trees = gbmGrid$trees[i], 
        cred = gbmGrid$cred_T1_V8[i], 
        rand = gbmGrid$rand_T1_V8[i], 
        eval_metric_final=eval_metric_final[i]),
      data.frame(varImp),
      data.frame(validation_collect)
    )
  }
  stopImplicitCluster()
  
  tune_results <- as.data.frame(collector[1])
  tune_results <- arrange(tune_results, desc(eval_metric_final))
  View(tune_results)
  
  varImp <- as.data.frame(collector[2])
  getGbmTuningParamPlots(tune_results, varImp)  # Visualize results
  
  varImp <- varImp %>% group_by(var) %>% summarise(avg_rel.inf=mean(rel.inf)) %>% arrange(desc(avg_rel.inf))
  varImp_gbm <<- varImp
  
  validation_gbm <<- as.data.frame(collector[3])
  
  if (alert) beep(10)
  
  return(tune_results)
}

runGbmTuning_par2 <- function(gbmGrid, data, model, var.monotone, distrib, cv_fold, numSteps=0, stepSize, alert=TRUE, cores=4) {
  
  eval_metric_final <- vector(length=nrow(gbmGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  varImp <- NULL
  
  registerDoParallel(cores)
  comb <- function(x, ...) {  
    mapply(rbind,x,...,SIMPLIFY=FALSE)
  }
  
  collector <- foreach (i=1:nrow(gbmGrid), .combine='comb', .multicombine=TRUE) %dopar% {
    
    eval_metric_cv <- replicate(numSteps + 1, vector(length=folds), simplify=FALSE)
    
    validation_collect <- NULL
    
    k <- 1
    
    # Train the model
    for (j in 1:folds) {      
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V5'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=0, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0)
      
      set.seed(1)
      
      gbmModel <- gbm(formula = model,
                      data = train,
                      distribution = distrib,
                      bag.fraction = gbmGrid$bag.fraction[i],
                      n.trees = gbmGrid$trees[i], 
                      shrinkage = gbmGrid$shrinkage[i],
                      interaction.depth = gbmGrid$interaction.depth[i],
                      n.minobsinnode = gbmGrid$n.minobsinnode[i],
                      var.monotone = var.monotone,
                      #verbose = TRUE, # won't print anyway with parallelization
                      n.cores = 1)
      
      varImp <- rbind(varImp, cbind(id=i, summary(gbmModel, plotit=FALSE)))
      
      # Score
      for (m in 1:(numSteps + 1)) {
        
        if (m > 1) {
          eval_metric_cv[[m]][[k]] <- ngini(validation[, all.vars(model)[1]], predict.gbm(gbmModel, validation, type='response', n.trees=gbmGrid$trees[i] - ((m-1) * stepSize)))
        }
        else {
          validation$score_gbm <- predict.gbm(gbmModel, validation, type='response', n.trees=gbmGrid$trees[i])
          validation_collect <- rbind(validation_collect, validation)
          
          eval_metric_cv[[1]][[k]] <- ngini(validation[, all.vars(model)[1]], validation$score_gbm)
          
        }
           
      }
     
      
      k <- k + 1
    }    
    
   
    gbmGrid_collect <- NULL
    
    for (m in 1:(numSteps + 1)) {
    
      gbmGrid_tmp <- data.frame(bag.fraction = gbmGrid$bag.fraction[i],  
                                 shrinkage = gbmGrid$shrinkage[i], 
                                 interaction.depth = gbmGrid$interaction.depth[i],  
                                 n.minobsinnode = gbmGrid$n.minobsinnode[i],  
                                 trees = gbmGrid$trees[i] - ((m-1) * stepSize), 
                                 eval_metric_final=mean(eval_metric_cv[[m]]))
      
      gbmGrid_collect <- rbind(gbmGrid_collect, gbmGrid_tmp)
    }
    
    
    list(
      gbmGrid_collect,
      data.frame(varImp),
      data.frame(validation_collect)
    )
  }
  stopImplicitCluster()
  
  tune_results <- as.data.frame(collector[1])
  tune_results <- arrange(tune_results, desc(eval_metric_final))
  View(tune_results)
  
  varImp <- as.data.frame(collector[2])
  getGbmTuningParamPlots(tune_results, varImp)  # Visualize results
  
  varImp <- varImp %>% group_by(var) %>% summarise(avg_rel.inf=mean(rel.inf)) %>% arrange(desc(avg_rel.inf))
  varImp_gbm <<- varImp
  
  validation_gbm <<- as.data.frame(collector[3])
  
  if (alert) beep(10)
  
  return(tune_results)
}


############################################################################################################
# Function to tune RF model parameters
runRfTuning <- function(rfGrid, data, model, cv_fold, alert=TRUE) {
  
  eval_metric_final <- vector(length=nrow(rfGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  varImp <- NULL
  
  for (i in 1:nrow(rfGrid)) {   
    
    cat("\n\n", "########## Start calibration of new model ##########", "\n", sep="");
    print(i)
    
    cat(' : ntree=', rfGrid$trees[i], 
        ' : mtry=', rfGrid$mtry[i], 
        ' : maxnodes=', rfGrid$maxnodes[i], 
        ' : nodesize=', rfGrid$nodesize[i],
        ' : sampsize=', rfGrid$sampsize[i],
        ' : replace=', rfGrid$replace[i],
        "\n\n", sep='')
    
    
    eval_metric_cv <- vector(length=folds)
    validation_collect <- NULL
    
    k <- 1
    
    # Train the model
    for (j in 1:folds) {      
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      set.seed(2)

      rfModel <<- randomForest(formula = model,
                               data = train,
                               ntree = rfGrid$trees[i],
                               mtry = rfGrid$mtry[i],
                               maxnodes = rfGrid$maxnodes[i],
                               nodesize = rfGrid$nodesize[i],
                               sampsize = rfGrid$sampsize[i],
                               replace = rfGrid$replace[i],
                               importance = TRUE)    
      
      imp <- importance(rfModel, type=1)  # either 1 or 2, specifying the type of importance measure (1=mean decrease in accuracy, 2=mean decrease in node impurity).
      featureImportance <- data.frame(var=row.names(imp), importance=imp[, 1])
      
      varImp <- rbind(varImp, cbind(id=i, featureImportance))
      
      # Score
      validation$score_rfo <- predict(rfModel, validation, type='response')
      validation_collect <- rbind(validation_collect, validation)
      
      eval_metric_cv[k] <- ngini(validation[, all.vars(model)[1]], validation$score_rfo)
      
      k <- k + 1
    }
    
    print(importance(rfModel))
    eval_metric_final[i] <- mean(eval_metric_cv)
    cat('\nEval. metric: ', eval_metric_final[i], sep='')
    
    tmp_tune_results <<- arrange(cbind(rfGrid, eval_metric_final), desc(eval_metric_final))  # For early stoppages
  }
  
  tune_results <- arrange(cbind(rfGrid, eval_metric_final), desc(eval_metric_final))
  View(tune_results)
  
  getRfTuningParamPlots(tune_results, varImp)  # Visualize results
  
  varImp <- varImp %>% group_by(var) %>% summarise(avg_importance=mean(importance)) %>% arrange(desc(avg_importance))
  varImp_rfo <<- varImp
  
  validation_rfo <<- validation_collect
  
  if (alert) beep(10)
  return(tune_results)
}


runRfTuning_par1 <- function(rfGrid, data, model, cv_fold, alert=TRUE, cores=4) {
  
  eval_metric_final <- vector(length=nrow(rfGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  varImp <- NULL
  
  registerDoParallel(cores)
  comb <- function(x, ...) {  
    mapply(rbind,x,...,SIMPLIFY=FALSE)
  }
  
  collector <- foreach (i=1:nrow(rfGrid), .combine='comb', .multicombine=TRUE) %dopar% {
    eval_metric_cv <- foreach (j=1:folds, .combine='c') %do% {
      
      validation_collect <- NULL
      
      # Train the model
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=5, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=20, rand=0.2)
      
      set.seed(2)
      
      rfModel <<- randomForest(formula = model,
                               data = train,
                               ntree = rfGrid$trees[i],
                               mtry = rfGrid$mtry[i],
                               maxnodes = rfGrid$maxnodes[i],
                               nodesize = rfGrid$nodesize[i],
                               sampsize = rfGrid$sampsize[i],
                               replace = rfGrid$replace[i],
                               importance = TRUE)    
      
      imp <- importance(rfModel, type=1)  # either 1 or 2, specifying the type of importance measure (1=mean decrease in accuracy, 2=mean decrease in node impurity).
      featureImportance <- data.frame(var=row.names(imp), importance=imp[, 1])
      varImp <- rbind(varImp, cbind(id=i, featureImportance))
      
      # Score
      validation$score_rfo <- predict(rfModel, validation, type='response')
      validation_collect <- rbind(validation_collect, validation)
      
      ngini(validation[, all.vars(model)[1]], validation$score_rfo)
    }    
    
    eval_metric_final[i] <- mean(eval_metric_cv)
    
    list(
      data.frame(trees = rfGrid$trees[i],  
                 mtry = rfGrid$mtry[i], 
                 maxnodes = rfGrid$maxnodes[i],  
                 nodesize = rfGrid$nodesize[i],
                 sampsize = rfGrid$sampsize[i],
                 replace = rfGrid$replace[i],
                 eval_metric_final=eval_metric_final[i]),
      data.frame(varImp),
      data.frame(validation_collect)
    )
  }
  stopImplicitCluster()
  
  tune_results <- as.data.frame(collector[1])
  tune_results <- arrange(tune_results, desc(eval_metric_final))
  View(tune_results)
  
   
  varImp <- as.data.frame(collector[2])
  getRfTuningParamPlots(tune_results, varImp)  # Visualize results
  
  varImp <- varImp %>% group_by(var) %>% summarise(avg_importance=mean(importance)) %>% arrange(desc(avg_importance))
  varImp_rfo <<- varImp
  
  validation_rfo <<- as.data.frame(collector[3])

  if (alert) beep(10)
  
  return(tune_results)
}


############################################################################################################
# Function to tune RF model parameters
runGntTuning <- function(gntGrid, data, model, family, cv_fold, alert=TRUE) {
  
  eval_metric_final <- vector(length=nrow(gntGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  
  for (i in 1:nrow(gntGrid)) {   
    
    cat("\n\n", "########## Start calibration of new model ##########", "\n", sep="");
    print(i)
    
    cat(' : alpha=', gntGrid$alpha[i], 
        ' : lambda.min.ratio=', gntGrid$lambda.min.ratio[i], 
        "\n\n", sep='')
    
    
    eval_metric_cv <- vector(length=folds)
    validation_collect <- NULL
    
    k <- 1
    
    # Train the model
    for (j in 1:folds) {      
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('username'), 'dropped', freq=TRUE, cred=5, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('username'), 'dropped', freq=TRUE, cred=5, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('course_id'), 'dropped', freq=TRUE, cred=40, rand=0.1)
      validation <- getOneWayVars_retTest(train, validation, c('course_id'), 'dropped', freq=TRUE, cred=40, rand=0.1)
      

      set.seed(4)
      
      glmnetModel <<- cv.glmnet(x = as.matrix(train[, all.vars(model)[-1]]),
                                y = as.matrix(train[, all.vars(model)[1]]),
                                family = family,  # gaussian, binomial, poisson, multinomial, cox, mgaussian
                                alpha = gntGrid$alpha[i],  # the elasticnet mixing parameter (penalty); alpha=1 is the lasso penalty, and alpha=0 the ridge penalty
                                lambda.min.ratio = gntGrid$lambda.min.ratio[i])  #Smallest value for lambda, as a fraction of lambda.max, the (data derived) entry
      #value (i.e. the smallest value for which all coefficients are zero). The default
      #depends on the sample size nobs relative to the number of variables nvars.
      #If nobs > nvars, the default is 0.0001, close to zero. If nobs < nvars,
      #the default is 0.01.
      #print(glmnetModel)
      
      # Score
      validation$score_gnt <- predict(glmnetModel, as.matrix(validation[, all.vars(model)[-1]]), s='lambda.min', type='response')[, 1]
      validation_collect <- rbind(validation_collect, validation)
      
      eval_metric_cv[k] <- ngini(validation[, all.vars(model)[1]], validation$score_gnt)
      
      k <- k + 1
    }
    
    eval_metric_final[i] <- mean(eval_metric_cv)
    cat('\nEval. metric: ', eval_metric_final[i], sep='')
    
    tmp_tune_results <<- arrange(cbind(gntGrid, eval_metric_final), desc(eval_metric_final))  # For early stoppages
  }
  
  tune_results <- arrange(cbind(gntGrid, eval_metric_final), desc(eval_metric_final))
  View(tune_results)
  

  validation_gnt <<- validation_collect
  
  if (alert) beep(10)
  
  return(tune_results)
}


runGntTuning_par <- function(gntGrid, data, model, family, cv_fold, alert=TRUE, cores=4) {
  
  eval_metric_final <- vector(length=nrow(gntGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  
  registerDoParallel(cores)
  comb <- function(x, ...) {  
    mapply(rbind,x,...,SIMPLIFY=FALSE)
  }
  
  collector <- foreach (i=1:nrow(gntGrid), .combine='comb', .multicombine=TRUE) %dopar% {
    eval_metric_cv <- foreach (j=1:folds, .combine='c') %do% {
      
      validation_collect <- NULL
      
      # Train the model
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V8'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V8'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V4'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V4'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T2_V12'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T2_V12'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      set.seed(4)
      
      glmnetModel <- cv.glmnet(x = as.matrix(train[, all.vars(model)[-1]]),
                                y = as.matrix(train[, all.vars(model)[1]]),
                                family = family,  # gaussian, binomial, poisson, multinomial, cox, mgaussian
                                alpha = gntGrid$alpha[i],  # the elasticnet mixing parameter (penalty); alpha=1 is the lasso penalty, and alpha=0 the ridge penalty
                                lambda.min.ratio = gntGrid$lambda.min.ratio[i])  #Smallest value for lambda, as a fraction of lambda.max, the (data derived) entry
      #value (i.e. the smallest value for which all coefficients are zero). The default
      #depends on the sample size nobs relative to the number of variables nvars.
      #If nobs > nvars, the default is 0.0001, close to zero. If nobs < nvars,
      #the default is 0.01.
      #print(glmnetModel)
      
      # Score
      validation$score_gnt <- predict(glmnetModel, as.matrix(validation[, all.vars(model)[-1]]), s='lambda.min', type='response')[, 1]
      validation_collect <- rbind(validation_collect, validation)
      
      ngini(validation[, all.vars(model)[1]], validation$score_gnt)
    }    
    
    eval_metric_final[i] <- mean(eval_metric_cv)
    
    list(
      data.frame(alpha = gntGrid$alpha[i],  
                 lambda.min.ratio = gntGrid$lambda.min.ratio[i], 
                 eval_metric_final=eval_metric_final[i]),
      data.frame(validation_collect)
    )
  }
  stopImplicitCluster()
  
  tune_results <- as.data.frame(collector[1])
  tune_results <- arrange(tune_results, desc(eval_metric_final))
  View(tune_results)
  
  validation_gnt <<- as.data.frame(collector[2])
  
  if (alert) beep(10)
  
  return(tune_results)
}


############################################################################################################
# Function to tune ET model parameters
runEtTuning <- function(etGrid, data, model, cv_fold, alert=TRUE) {
  
  eval_metric_final <- vector(length=nrow(etGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  
  
  for (i in 1:nrow(etGrid)) {   
    
    cat("\n\n", "########## Start calibration of new model ##########", "\n", sep="");
    print(i)
    
    cat(' : ntree=', etGrid$trees[i], 
        ' : mtry=', etGrid$mtry[i], 
        ' : nodesize=', etGrid$nodesize[i], 
        ' : numRandomCuts=', etGrid$numRandomCuts[i],
        ' : evenCuts=', etGrid$evenCuts[i],
        "\n\n", sep='')
    
    eval_metric_cv <- vector(length=folds)
    validation_collect <- NULL
    
    k <- 1
    
    # Train the model
    for (j in 1:folds) {      
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('username'), 'dropped', freq=TRUE, cred=5, rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('username'), 'dropped', freq=TRUE, cred=5, rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('course_id'), 'dropped', freq=TRUE, cred=40, rand=0.1)
      validation <- getOneWayVars_retTest(train, validation, c('course_id'), 'dropped', freq=TRUE, cred=40, rand=0.1)
      
      
      set.seed(3)
                
      etModel <<- extraTrees(x = train[, all.vars(model)[-1]],
                             y = train[, all.vars(model)[1]],
                             ntree = etGrid$trees[i],
                             mtry = etGrid$mtry[i],
                             nodesize = etGrid$nodesize[i],
                             numRandomCuts = etGrid$numRandomCuts[i],
                             evenCuts = etGrid$evenCuts[i],
                             #na.action = 'fuse',
                             numThreads = 2)     
      # Score
      validation$score_ext <- predict(etModel, validation[, all.vars(model)[-1]])
      validation_collect <- rbind(validation_collect, validation)
      
      eval_metric_cv[k]<- ngini(validation[, all.vars(model)[1]], validation$score_ext)
      
      k <- k + 1
      
    }

    eval_metric_final[i] <- mean(eval_metric_cv)
    cat('\nEval. metric: ', eval_metric_final[i], sep='')
    
    tmp_tune_results <<- arrange(cbind(etGrid, eval_metric_final), desc(eval_metric_final))  # For early stoppages
  }
  
  tune_results <- arrange(cbind(etGrid, eval_metric_final), desc(eval_metric_final))
  View(tune_results)
  
  getEtTuningParamPlots(tune_results)  # Visualize results
  
  validation_ext <<- validation_collect
  
  if (alert) beep(10)
  
  return(tune_results)
}


runEtTuning_par <- function(etGrid, data, model, cv_fold, alert=TRUE, cores=4) {
  
  eval_metric_final <- vector(length=nrow(etGrid))
  data$cv_var <- data[, cv_fold]
  folds <- max(data$cv_var)
  
  registerDoParallel(cores)
  comb <- function(x, ...) {  
    mapply(rbind,x,...,SIMPLIFY=FALSE)
  }
  
  collector <- foreach (i=1:nrow(etGrid), .combine='comb', .multicombine=TRUE) %dopar% {
    eval_metric_cv <- foreach (j=1:folds, .combine='c') %do% {
      
      validation_collect <- NULL
      
      # Train the model
      train <- subset(data, !(cv_var %in% j))
      validation <- subset(data, cv_var %in% j)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V8'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V8'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V16'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V11'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V12'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V4'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V4'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V15'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T1_V7'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      train <- getOneWayVars_retTrain(train, validation, c('T2_V12'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      validation <- getOneWayVars_retTest(train, validation, c('T2_V12'), 'Hazard', freq=TRUE, cred=gbmGrid$cred_T1_V8[i], rand=0)
      
      set.seed(3)
      
      etModel <<- extraTrees(x = train[, all.vars(model)[-1]],
                             y = train[, all.vars(model)[1]],
                             ntree = etGrid$trees[i],
                             mtry = etGrid$mtry[i],
                             nodesize = etGrid$nodesize[i],
                             numRandomCuts = etGrid$numRandomCuts[i],
                             evenCuts = etGrid$evenCuts[i],
                             #na.action = 'fuse',
                             numThreads = 1)     
      # Score
      validation$score_ext <- predict(etModel, validation[, all.vars(model)[-1]])
      validation_collect <- rbind(validation_collect, validation) 
      
      ngini(validation[, all.vars(model)[1]], validation$score_ext)
    }    
    
    eval_metric_final[i] <- mean(eval_metric_cv)
    
    list(
      data.frame(trees = etGrid$trees[i],  
                 mtry = etGrid$mtry[i], 
                 nodesize = etGrid$nodesize[i],  
                 numRandomCuts = etGrid$numRandomCuts[i], 
                 evenCuts = etGrid$evenCuts[i],  
                 eval_metric_final=eval_metric_final[i]),
      data.frame(validation_collect)
    )
  }
  stopImplicitCluster()
  
  tune_results <- as.data.frame(collector[1])
  tune_results <- arrange(tune_results, desc(eval_metric_final))
  View(tune_results)
  
  
  getEtTuningParamPlots(tune_results)  # Visualize results
  
  validation_ext <<- as.data.frame(collector[2])
  
  if (alert) beep(10)
  
  return(tune_results)
}


############################################################################################################
# Plots results for Xgboost tuning (requires ggplot2 & dplyr)
getXgbTuningParamPlots <- function(data, data2) { 
  
  #data$eval.metric <- data$eval_metric_final
  data$eval.metric <- data$eval_metric_final  
  
  # See which trees performed best
  summNround <- data %>%
    group_by(nround) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Nround <- ggplot() +
    geom_point(data=data, aes(x=nround, y=eval.metric)) +
    geom_point(data=summNround, aes(x=nround, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$nround)) +
    ggtitle('Nround vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which subsample performed best
  summSubsample <- data %>%
    group_by(subsample) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Subsample <- ggplot() +
    geom_point(data=data, aes(x=subsample, y=eval.metric)) +
    geom_point(data=summSubsample, aes(x=subsample, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$subsample)) +
    ggtitle('Subsample vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which eta performed best
  summEta <- data %>%
    group_by(eta) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Eta <- ggplot() +
    geom_point(data=data, aes(x=eta, y=eval.metric)) +
    geom_point(data=summEta, aes(x=eta, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$eta)) +
    ggtitle('Eta vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which max.depth performed best
  summMax.depth <- data %>%
    group_by(max.depth) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Max.depth <- ggplot() +
    geom_point(data=data, aes(x=max.depth, y=eval.metric)) +
    geom_point(data=summMax.depth, aes(x=max.depth, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$max.depth)) +
    ggtitle('Max.depth vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which min.child.weight performed best
  summMin.child.weight <- data %>%
    group_by(min.child.weight) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Min.child.weight <- ggplot() +
    geom_point(data=data, aes(x=min.child.weight, y=eval.metric)) +
    geom_point(data=summMin.child.weight, aes(x=min.child.weight, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$min.child.weight)) +
    ggtitle('Min.child.weight vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which colsample_bytree performed best
  summColsample_bytree <- data %>%
    group_by(colsample_bytree) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Colsample_bytree <- ggplot() +
    geom_point(data=data, aes(x=colsample_bytree, y=eval.metric)) +
    geom_point(data=summColsample_bytree, aes(x=colsample_bytree, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$colsample_bytree)) +
    ggtitle('Colsample_bytree vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # Variable importance
  summImportance <<- data2 %>%
    group_by(var) %>%
    summarise(
      medImportance = median(importance)
    )
  
  Importance <- ggplot() +  
    geom_point(data=data2, aes(x=var, y=importance)) + 
    geom_point(data=summImportance, aes(x=var, y=medImportance), size=5) +
    ggtitle('Relative infleunce\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14, angle=90)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
   
  data2 <- data2 %>% group_by(var) %>% summarise(avg_importance=mean(importance))
  featureImportance <- ggplot(data2, aes(x=reorder(var, avg_importance), y=avg_importance)) +
    geom_bar(stat="identity", fill="#53cfff") +
    coord_flip() + 
    theme_light(base_size=20) +
    xlab("Features") +
    ylab("") + 
    ggtitle("Xgboost Avg Feature Importance\n") +
    theme(plot.title=element_text(size=18)) 
  
  #multiplot(Nround, Subsample, Max.depth, cols=2) 
  multiplot(Nround, Subsample, Eta, Max.depth, cols=2) 
  multiplot(Min.child.weight, Colsample_bytree, Importance, cols=2) 
  multiplot(featureImportance, cols=1)
}


############################################################################################################
# Plots results for GBM tuning (requires ggplot2 & dplyr)
getGbmTuningParamPlots <- function(data, data2) {  
  
  data$eval.metric <- data$eval_metric_final
  
  # See which shrinkage performed best
  summShrinkage <- data %>%
    group_by(shrinkage) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    ) 
  
  shrinkage <- ggplot() +
    geom_point(data=data, aes(x=shrinkage, y=eval.metric)) + 
    geom_point(data=summShrinkage, aes(x=shrinkage, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$shrinkage)) +
    ggtitle('Shrinkage vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which tree performed best
  summTrees <- data %>%
    group_by(trees) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Trees <- ggplot() +
    geom_point(data=data, aes(x=trees, y=eval.metric)) +
    geom_point(data=summTrees, aes(x=trees, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$trees)) +
    ggtitle('Trees vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which interaction depth performed best
  summIntDepth <- data %>%
    group_by(interaction.depth) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  intDepth <- ggplot() +
    geom_point(data=data, aes(x=interaction.depth, y=eval.metric)) +
    geom_point(data=summIntDepth, aes(x=interaction.depth, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$interaction.depth)) +
    ggtitle('Interaction depth vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which min terminal nodes performed best
  summMinobs <- data %>%
    group_by(n.minobsinnode) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  obsNodes <- ggplot() +  
    geom_point(data=data, aes(x=n.minobsinnode, y=eval.metric)) + 
    geom_point(data=summMinobs, aes(x=n.minobsinnode, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$n.minobsinnode)) + 
    ggtitle('Min Obs in Terminal Nodes vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which bag fraction performed best
  summBagfraction <- data %>%
    group_by(bag.fraction) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  bagFraction <- ggplot() +  
    geom_point(data=data, aes(x=bag.fraction, y=eval.metric)) + 
    geom_point(data=summBagfraction, aes(x=bag.fraction, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$bag.fraction)) + 
    ggtitle('Bag fraction vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See relative influence of features
    summRelinf <- data2 %>%
    group_by(var) %>%
    summarise(
      medRelInf = median(rel.inf)
    )
  
  influence <- ggplot() +  
    geom_point(data=data2, aes(x=var, y=rel.inf)) + 
    geom_point(data=summRelinf, aes(x=var, y=medRelInf), size=5) +
    ggtitle('Relative infleunce\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14, angle=90)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
data2 <- data2 %>% group_by(var) %>% summarise(avgRelInf=mean(rel.inf))
featureImportance <- ggplot(data2, aes(x=reorder(var, avgRelInf), y=avgRelInf)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("Features") +
  ylab("") + 
  ggtitle("GBM Avg Relative Influence\n") +
  theme(plot.title=element_text(size=18))
  
  multiplot(shrinkage, Trees, bagFraction, cols=2) 
  multiplot(intDepth, obsNodes, influence, cols=2)
  multiplot(featureImportance, cols=1)
}


############################################################################################################
# Plots results for Random Forest tuning (requires ggplot2 & dplyr)
getRfTuningParamPlots <- function(data, data2) { 
  #data$eval.metric <- data$eval_metric_final
  data$eval.metric <- data$eval_metric_final
  
  # See which trees performed best
  summTrees <- data %>%
    group_by(trees) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Trees <- ggplot() +
    geom_point(data=data, aes(x=trees, y=eval.metric)) +
    geom_point(data=summTrees, aes(x=trees, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$trees)) +
    ggtitle('Trees vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which mtry performed best
  summMtry <- data %>%
    group_by(mtry) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Mtry <- ggplot() +
    geom_point(data=data, aes(x=mtry, y=eval.metric)) +
    geom_point(data=summMtry, aes(x=mtry, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$mtry)) +
    ggtitle('Mtry vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which nodesize performed best
  summNodesize <- data %>%
    group_by(nodesize) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Nodesize <- ggplot() +
    geom_point(data=data, aes(x=nodesize, y=eval.metric)) +
    geom_point(data=summNodesize, aes(x=nodesize, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$nodesize)) +
    ggtitle('Nodesize vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which maxnodes performed best
  summMaxnodes <- data %>%
    group_by(maxnodes) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Maxnodes <- ggplot() +
    geom_point(data=data, aes(x=maxnodes, y=eval.metric)) +
    geom_point(data=summMaxnodes, aes(x=maxnodes, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$maxnodes)) +
    ggtitle('Maxnodes vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which sampsize performed best
  summSampsize <- data %>%
    group_by(sampsize) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Sampsize <- ggplot() +
    geom_point(data=data, aes(x=sampsize, y=eval.metric)) +
    geom_point(data=summSampsize, aes(x=sampsize, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$sampsize)) +
    ggtitle('Sampsize vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which replace performed best
  data$replace2 <- ifelse(data$replace == TRUE, 1, 0)
  
  summReplace <- data %>%
    group_by(replace2) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Replace <- ggplot() +
    geom_point(data=data, aes(x=replace2, y=eval.metric)) +
    geom_point(data=summReplace, aes(x=replace2, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$replace2)) +
    ggtitle('Replace vs. eval metric\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # Variable importance
  summImportance <<- data2 %>%
    group_by(var) %>%
    summarise(
      medImportance = median(importance)
    )
  
  Importance <- ggplot() +  
    geom_point(data=data2, aes(x=var, y=importance)) + 
    geom_point(data=summImportance, aes(x=var, y=medImportance), size=5) +
    ggtitle('Relative infleunce\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14, angle=90)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))

  data2 <- data2 %>% group_by(var) %>% summarise(avg_importance=mean(importance))
  featureImportance <- ggplot(data2, aes(x=reorder(var, avg_importance), y=avg_importance)) +
    geom_bar(stat="identity", fill="#53cfff") +
    coord_flip() + 
    theme_light(base_size=20) +
    xlab("Features") +
    ylab("") + 
    ggtitle("Random Forest Avg Feature Importance\n") +
    theme(plot.title=element_text(size=18))
  
  multiplot(Trees, Mtry, Nodesize, Maxnodes, cols=2) 
  multiplot(Sampsize, Replace, Importance, cols=2)
  multiplot(featureImportance, cols=1)
}


############################################################################################################
# Plots results for Extra Trees tuning (requires ggplot2 & dplyr)
getEtTuningParamPlots <- function(data) { 
  
  #data$eval.metric <- data$eval_metric_final
  data$eval.metric <- data$eval_metric_final
  
  
  # See which trees performed best
  summTrees <- data %>%
    group_by(trees) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Trees <- ggplot() +
    geom_point(data=data, aes(x=trees, y=eval.metric)) +
    geom_point(data=summTrees, aes(x=trees, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$trees)) +
    ggtitle('Trees vs. log loss\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which mtry performed best
  summMtry <- data %>%
    group_by(mtry) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Mtry <- ggplot() +
    geom_point(data=data, aes(x=mtry, y=eval.metric)) +
    geom_point(data=summMtry, aes(x=mtry, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$mtry)) +
    ggtitle('Mtry vs. log loss\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which nodesize performed best
  summNodesize <- data %>%
    group_by(nodesize) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  Nodesize <- ggplot() +
    geom_point(data=data, aes(x=nodesize, y=eval.metric)) +
    geom_point(data=summNodesize, aes(x=nodesize, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$nodesize)) +
    ggtitle('Nodesize vs. log loss\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
  
  # See which numRandomCuts performed best
  summNumRandomCuts <- data %>%
    group_by(numRandomCuts) %>%
    summarise(
      medEvalMetric = median(eval.metric)
    )
  
  NumRandomCuts <- ggplot() +
    geom_point(data=data, aes(x=numRandomCuts, y=eval.metric)) +
    geom_point(data=summNumRandomCuts, aes(x=numRandomCuts, y=medEvalMetric), size=5) +
    scale_x_continuous(breaks=unique(data$numRandomCuts)) +
    ggtitle('NumRandomCuts vs. log loss\n(Median values in bold)') +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18, angle=90)) +
    theme(axis.text.x = element_text(size=14)) + 
    theme(axis.text.y = element_text(size=14)) +
    theme(plot.title = element_text(size=20))
    
 
  multiplot(Trees, Mtry, Nodesize, NumRandomCuts, cols=2) 
}


############################################################################################################
# Plot random forest feature importance
plotFeatureImportance <- function(modelObj, method, sparse_matrix) {
  
  if (method == 'xgb') { 
    featureImportance <- xgb.importance(sparse_matrix@Dimnames[[2]], model=modelObj)
    print(featureImportance)
    graph <- xgb.plot.importance(importance_matrix=featureImportance)
  }
  else if (method == 'gbm') { 
  featureImportance <- summary(gbmModel, plotit=FALSE)
  
  graph <- ggplot(featureImportance, aes(x=reorder(var, rel.inf), y=rel.inf)) +
    geom_bar(stat="identity", fill="#53cfff") +
    coord_flip() + 
    theme_light(base_size=20) +
    xlab("Features") +
    ylab("") + 
    ggtitle("GBM Avg Relative Influence\n") +
    theme(plot.title=element_text(size=18))
  }
  else if (method == 'rfo') {
    imp <- importance(modelObj, type=1)  # either 1 or 2, specifying the type of importance measure (1=mean decrease in accuracy, 2=mean decrease in node impurity).
    featureImportance <- data.frame(feature=row.names(imp), importance=imp[, 1])
    
    graph <- ggplot(featureImportance, aes(x=reorder(feature, importance), y=importance)) +
      geom_bar(stat="identity", fill="#53cfff") +
      coord_flip() + 
      theme_light(base_size=20) +
      xlab("Features") +
      ylab("") + 
      ggtitle("Random Forest Feature Importance\n") +
      theme(plot.title=element_text(size=18))
  }
  
  multiplot(graph, cols=1)
}


############################################################################################################
# Exports submission file and final test dataset
exportSubmission <- function(data, num, cols, replace) {
  # requires dplyr
  
  sub <- data[, cols]
  names(sub) <- replace
  
  write.csv(sub, paste('submissions/submission', num, '.csv', sep=''), row.names=FALSE)  
}

