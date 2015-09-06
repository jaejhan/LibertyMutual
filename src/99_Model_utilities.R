# Exports submission file and final test dataset ------------------------------
getModelPerformance <- function(data, actual, predicted, bins, output=FALSE) {
  
  data$aa <- data[, actual]
  data$pp <- data[, predicted]
    
  #   auc <- getAUCSummary(data, predicted='score_final', actual='actual')
  #   cat('\nAUC = ', auc, sep='') 
  
#   gini <- getGini(data$aa, data$pp)
#   cat('\nGini = ', gini, sep='') 
  
  ngini <- ngini(data$aa, data$pp)
  cat('\nNorm. Gini = ', ngini, sep='') 

  logloss <- getRMSE(data$aa, data$pp)
  cat('\nLog loss = ', logloss, sep='') 
  
#   RMSE <- getRMSE(data$aa, data$pp)
#   cat('\nRMSE = ', RMSE, sep='') 
#   
#   MAE <- getMAE(data$aa, data$pp)
#   cat('\nMAE = ', MAE, sep='') 

if (output) {
  summary_gbm <- data %>% group_by(season) %>% summarise(Ngini=getNormalizedGini(aa, score_gbm), Logloss=getLogLoss(aa, score_gbm)) %>% mutate(Model='GBM') %>% ungroup()
  summary_glm <- data %>% group_by(season) %>% summarise(Ngini=getNormalizedGini(aa, score_glm), Logloss=getLogLoss(aa, score_glm)) %>% mutate(Model='GLM') %>% ungroup()
  summary_rfo <- data %>% group_by(season) %>% summarise(Ngini=getNormalizedGini(aa, score_rfo), Logloss=getLogLoss(aa, score_rfo)) %>% mutate(Model='RFO') %>% ungroup()
  
  summary_table <- rbind(summary_gbm, summary_glm, summary_rfo)
  summary_final <- data %>% group_by(season) %>% summarise(Logloss=getLogLoss(aa, pp)) %>% ungroup()
 
  summary_table$Logloss_dec3 <- round(summary_table$Logloss, 3)
  summary_final$Logloss_dec3 <- round(summary_final$Logloss, 3)
  
  ggplot(summary_table, aes(x=season, y=Logloss, colour=Model)) +
    geom_point(size=3.0) +
    geom_path(size=1.0) +
    #geom_text(aes(x=season, label=Logloss_dec3, vjust=-0.5)) +
    ggtitle('Log Loss') +
    xlab('Season') +
    ylab('') +
    scale_x_continuous(breaks=2002:2014) +
    scale_y_continuous(limits=c(0, 1)) +
    scale_color_brewer(palette="Set1") +
    theme(plot.title = element_text(size=12)) +
    theme(axis.title.x = element_text(size=10)) +
    theme(axis.title.y = element_text(size=10)) +
    theme(axis.text.x = element_text(size=9)) +
    theme(axis.text.y = element_text(size=9))
  
  ggplot(summary_final, aes(x=season, y=Logloss)) +
    geom_point(size=3.0) +
    geom_path(size=1.0) +
    geom_text(aes(x=season, label=Logloss_dec3, vjust=-0.5)) +
    ggtitle('Log Loss') +
    xlab('Season') +
    ylab('') +
    scale_x_continuous(breaks=2002:2014) +
    scale_y_continuous(limits=c(0, 1)) +
    scale_color_brewer(palette="Set1") +
    theme(plot.title = element_text(size=12)) +
    theme(axis.title.x = element_text(size=10)) +
    theme(axis.title.y = element_text(size=10)) +
    theme(axis.text.x = element_text(size=9)) +
    theme(axis.text.y = element_text(size=9))
  
}
  
  #plotLiftChart(data, response='aa', predicted='pp', numBins=bins, cap=FALSE, cap_pct=0)
}


# Calculates the Gini evaluation metric -------------------------------------------
# "NormalizedGini" is the other half of the metric. This function does most of the work, though
SumModelGini <- function(solution, submission) {
  df <- data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE), ]
  
  df$random <- (1:nrow(df)) / nrow(df)
  
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  
  return(sum(df$Gini))
}

ngini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

# Function to be called within xgboost.train
evalgini <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- ngini(as.numeric(labels), as.numeric(preds))
  return(list(metric = "Gini", value = err))
}


# Calculates the RMSE (root mean squared error) evaluation metric --------------------------
getRMSE <- function(actual, predicted) {
  
  RMSE <- sqrt(mean((actual - predicted)^2))
  
  return(RMSE)
}


# Calculates the MAE (Mean Absolute Error) evaluation metric ---------------------------------
getMAE <- function(actual, predicted) {
  
  MAE <- mean(abs(actual - predicted))
  
  return(MAE)
}


# Calculates the log loss evaluation metric ---------------------------------
getLogLoss <- function(actual, predicted) {

  logloss <- (-1/length(actual)) * sum((actual * log(predicted)) + ((1-actual)*log(1-predicted)))
  
  return(logloss)
}


# Calculates the log loss evaluation metric ---------------------------------
getLogLoss2 <- function(actual, predicted) {
  
  logloss <- (-1/length(actual)) * sum(actual * log(predicted))
  
  return(logloss)
}


# Calculates the AUC evaluation metric ---------------------------------
getAUCSummary <- function(data, predicted, actual, output=TRUE) {
  yhat <- data[, predicted]
  y <- data[, actual]
  
  rocObj <- roc(y, yhat, na.rm=TRUE)
  
  if (output) {
    plot(rocObj, print.auc=TRUE, print.thres=FALSE, reuse.auc=TRUE, col="red")
  }
  
  return(rocObj$auc) 
}


# Plot Actuals vs. Predicted scatter ---------------------------------
plotActualvPredicted <- function(data, actual, predicted, title='None', xlab='X-label', ylab='Y-label', xlim=NULL, ylim=NULL) {
  data$aa <- data[, actual]
  data$pp <- data[, predicted]
  
  ggplot(data, aes(x=aa, y=pp)) +
    geom_point(alpha=0.33) +
    geom_abline(intercept=0, slope=1, colour = '#0072B2') + 
    coord_fixed() +
    ggtitle(title) +
    xlab(xlab) +
    ylab(ylab) +
    scale_x_continuous(limits=xlim, labels=comma) +
    scale_y_continuous(limits=ylim, labels=comma) +
    theme(plot.title = element_text(size=14)) +
    theme(axis.title.x = element_text(size=12)) +
    theme(axis.title.y = element_text(size=12)) +
    theme(axis.text.x = element_text(size=10, angle=90)) +
    theme(axis.text.y = element_text(size=10))
}

############################################################################################################
# Plot Residual plots
plotActualvResiduals <- function(data, actual, predicted, title='None', xlab='X-label', ylab='Residual (Predicted - Actual)', xlim=NULL, ylim=NULL) {
  data$aa <- data[, actual]
  data$rr <- data[, predicted] - data[, actual] 
  
  ggplot(data, aes(x=aa, y=rr)) +
    geom_point(alpha=0.33) +
    geom_abline(intercept=0, slope=0, colour = '#0072B2') + 
    ggtitle(title) +
    xlab(xlab) +
    ylab(ylab) +
    scale_x_continuous(limits=xlim, labels=comma) +
    scale_y_continuous(limits=ylim, labels=comma) +
    theme(plot.title = element_text(size=14)) +
    theme(axis.title.x = element_text(size=12)) +
    theme(axis.title.y = element_text(size=12)) +
    theme(axis.text.x = element_text(size=10, angle=90)) +
    theme(axis.text.y = element_text(size=10))
}


# Plot Residuals By Regressors ---------------------------------
plotResidualsByRegressors <- function(data, actual, predicted, regressors, title='None', xlab='X-label', ylab='Residual (Predicted - Actual)', xlim=NULL, ylim=NULL) {
  data$rr <- data[, predicted] - data[, actual] 
  
  data$ii <- data[ , regressors]
  
  if (is.numeric(data$ii)) {
    ggplot(data, aes(x=ii, y=rr)) +
      geom_point(alpha=0.33) +
      geom_abline(intercept=0, slope=0, colour = '#0072B2') + 
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(limits=xlim, labels=comma) +
      scale_y_continuous(limits=ylim, labels=comma) +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }

  else if (is.character(data$ii) | is.factor(data$ii)) {
     ggplot(data, aes(x=ii, y=rr)) +
        geom_point(alpha=0.33) +
        geom_abline(intercept=0, slope=0, colour = '#0072B2') + 
        ggtitle(title) +
        xlab(xlab) +
        ylab(ylab) +
        scale_y_continuous(limits=ylim, labels=comma) +
        theme(plot.title = element_text(size=14)) +
        theme(axis.title.x = element_text(size=12)) +
        theme(axis.title.y = element_text(size=12)) +
        theme(axis.text.x = element_text(size=10, angle=90)) +
        theme(axis.text.y = element_text(size=10))    
  }

}


# Plot Actual vs. Predictors By Regressors ---------------------------------
plotActualvPredictedByRegressors <- function(data, actual, predicted, regressors, bins=NULL, title='None', xlab='X-label', ylab='Total Claims', xlim=NULL, ylim=NULL) {
  
  data$aa <- data[ , actual]
  data$pp <- data[ , predicted]
  data$ii <- data[ , regressors]
  
  if (is.numeric(data$ii)) {
    
    data$ii_bin <- binVariable(data$ii, bins=bins)
    summary_df1 <- data %>% group_by(ii_bin) %>% summarise(total_loss=sum(aa)) %>% mutate(Group='Actual') %>% ungroup()
    summary_df2 <- data %>% group_by(ii_bin) %>% summarise(total_loss=sum(pp)) %>% mutate(Group='Predicted') %>% ungroup()
    summary_df <- rbind(summary_df1, summary_df2)

    ggplot(summary_df, aes(x=ii_bin, y=total_loss, fill=Group)) +
      geom_bar(position='dodge', stat='identity') +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(breaks=1:bins, limits=xlim) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10)) +
      theme(axis.text.y = element_text(size=10))
  }
  
  else if (is.character(data$ii) | is.factor(data$ii)) {
    summary_df1 <- data %>% group_by(ii) %>% summarise(total_loss=sum(aa)) %>% mutate(Group='Actual') %>% ungroup()
    summary_df2 <- data %>% group_by(ii) %>% summarise(total_loss=sum(pp)) %>% mutate(Group='Predicted') %>% ungroup()
    summary_df <- rbind(summary_df1, summary_df2)
    
    ggplot(summary_df, aes(x=ii, y=total_loss, fill=Group)) +
      geom_bar(position='dodge', stat='identity') +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }
  
}


# Plot % Difference By Regressors ---------------------------------
plotPctDiffByRegressors <- function(data, actual, predicted, regressors, bins=NULL, title='None', xlab='X-label', ylab='% Difference', xlim=NULL, ylim=NULL) {
  
  data$aa <- data[ , actual]
  data$pp <- data[ , predicted]
  data$ii <- data[ , regressors]
  
  if (is.numeric(data$ii)) {
    
    data$ii_bin <- binVariable(data$ii, bins=bins)
    summary_df <- data %>% group_by(ii_bin) %>% summarise(total_actual_loss=sum(aa), total_predicted_loss=sum(pp), pct_diff=(total_predicted_loss - total_actual_loss)/total_actual_loss) %>% mutate(Group='Actual') %>% ungroup()
    summary_df$pct_diff_dec <- round(summary_df$pct_diff, 2)
    
    ggplot(summary_df, aes(x=ii_bin, y=pct_diff)) +
      geom_bar(stat='identity', fill='#0072B2') +
      geom_text(aes(x=ii_bin, label=pct_diff_dec, vjust=-0.5)) +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(breaks=1:bins, limits=xlim) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10)) +
      theme(axis.text.y = element_text(size=10))
  }
  
  else if (is.character(data$ii) | is.factor(data$ii)) {
    summary_df <- data %>% group_by(ii) %>% summarise(total_actual_loss=sum(aa), total_predicted_loss=sum(pp), pct_diff=(total_predicted_loss - total_actual_loss)/total_actual_loss) %>% mutate(Group='Actual') %>% ungroup()
    summary_df$pct_diff_dec <- round(summary_df$pct_diff, 2)
    
    ggplot(summary_df, aes(x=ii, y=pct_diff)) +
      geom_bar(stat='identity', fill='#0072B2') +
      geom_text(aes(x=ii, label=pct_diff_dec, vjust=-0.5)) +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }
  
}


# Plot Boxplots By Regressors ---------------------------------
plotClaimsBoxplotsByRegressors <- function(data, claims, regressors, bins=NULL, title='None', xlab='X-label', ylab='% Difference', xlim=NULL, ylim=NULL) {
  
  data$cc <- data[ , claims]
  data$ii <- data[ , regressors]
  
  if (is.numeric(data$ii)) {
    
    data$ii_bin <- binVariable(data$ii, bins=bins)
    data$ii_bin <- as.factor(data$ii_bin)
    
    ggplot(data, aes(x=ii_bin, y=cc)) +
      geom_boxplot() +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10)) +
      theme(axis.text.y = element_text(size=10))
  }
  
  else if (is.character(data$ii) | is.factor(data$ii)) {
    
    ggplot(data, aes(x=ii, y=cc)) +
      geom_boxplot() +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }
  
}


# Plot Boxplots By Regressors ---------------------------------
plotScatter <- function(data, x, y, title='None', xlab='X-label', ylab='% Difference', xlim=NULL, ylim=NULL) {
  
  data$xx <- data[ , x]
  data$yy <- data[ , y]
  
  if (is.numeric(data$xx)) {
    
    ggplot(data, aes(x=xx, y=yy)) +
      geom_point(alpha=0.33) +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(limits=xlim, labels=comma) +
      scale_y_continuous(limits=ylim, labels=comma) +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
    
    
  }
  
  else if (is.character(data$ii) | is.factor(data$ii)) {
   
    ggplot(data, aes(yy)) +
      geom_point(alpha=0.33, position = "jitter") +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(limits=xlim, labels=comma) +
      scale_y_continuous(limits=ylim, labels=comma) +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }
  
}


# Plots lift charts ---------------------------------
plotLiftChart <- function(data, response, predicted, numBins, cap=TRUE, cap_pct=0) {
  # Requires dplyr
  data$response <- data[, response]  
  data$predicted <- data[, predicted]
  
  # Cap score if cap=TRUE
  if (cap) { 
    data$predicted[data$predicted > quantile(data$predicted, 1-cap_pct)] = quantile(data$predicted, 1-cap_pct);  # Cap predicted loss ratios
  }
  
  # Bin the data into 10 bins
  data$scoreBin <- ntile(data$predicted, numBins)
  #data$scoreBin = as.numeric(cut2(data$score_final, g=numBins))
  
  # Calculate predicted and realized probability data
  predicted <- data %>%
    group_by(scoreBin) %>%
    summarise(stat = mean(predicted),
              description=' Predicted    '
    )
  
  actual <- data %>%
    group_by(scoreBin) %>%
    summarise(stat = mean(response),
              description=' Actual    '
    )
  
  liftChart <<- rbind(predicted, actual)
  
  # Graph lift chart
  ggplot(liftChart, aes(x=scoreBin, y=stat, fill=description)) +
    geom_bar(position='dodge', stat='identity') +
    scale_fill_brewer(palette="Set1") +
    ggtitle('Lift Chart') +
    xlab('Predicted score bins') +
    ylab('Response') +
    scale_x_continuous(breaks=1:numBins, limits=c(0, numBins+1)) +
    scale_y_continuous(labels=comma) +
    theme_bw() +
    theme(plot.title = element_text(size=20)) +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18)) +
    theme(axis.text.x = element_text(size=14)) +
    theme(axis.text.y = element_text(size=14)) +
    theme(legend.title = element_blank()) +
    theme(legend.text = element_text(size=18)) +
    theme(legend.position='bottom', legend.box='horizontal')
}


# Returns common descriptive stats for datasets ---------------------------------
getSummaryStats <- function(data, varList, yvar, sortMissing=FALSE, export=FALSE) {
  dataName <- deparse(substitute(data))
  
  if (yvar == 'none') {
    data$y <- NA  
  } else {
    data$y <- data[, yvar]
  }
  
  # Gives you percentiles, mean, max, std.dev for list of vectors
  # Required packages: dplyr
  len <- length(varList)
  stats = data.frame(variable = vector(length=len),
                     data_type = vector(length=len),
                     N = vector(length=len),
                     missing = vector(length=len),
                     missingPct = vector(length=len),
                     uniqueVals = vector(length=len),
                     corr = vector(length=len),
                     corr_rank = vector(length=len),
                     maxInfCoef = vector(length=len),
                     min = vector(length=len),
                     pct01 = vector(length=len),
                     pct02 = vector(length=len),
                     pct05 = vector(length=len),
                     pct25 = vector(length=len),
                     pct50 = vector(length=len),
                     pct75 = vector(length=len),
                     pct95 = vector(length=len),
                     pct98 = vector(length=len),
                     pct99 = vector(length=len),
                     max = vector(length=len),
                     mean = vector(length=len),
                     std.dev = vector(length=len))
  
  for (i in 1:len) {
    var = data[, varList[i]]
    response = data[, 'y']
    stats$variable[i] = varList[i] 
    stats$data_type[i] = class(var) 
    
    if (is.numeric(var)) {
      stats$N[i] = length(which(!is.na(var)))
      stats$missing[i] = length(which(is.na(var)))
      stats$missingPct[i] = length(which(is.na(var))) / (length(which(!is.na(var))) + length(which(is.na(var))))
      stats$uniqueVals[i] = length(unique(var))
      stats$corr[i] = cor(var, response, method='pearson')  # standard correlation
      stats$corr_rank[i] = cor(var, response, method='spearman')  # rank correlation
      stats$maxInfCoef[i] = mine(var, response)[[1]]  # maximal information coefficient
      stats$min[i] = min(var, na.rm=TRUE)
      stats$pct01[i] = quantile(var, 0.01, na.rm=TRUE)
      stats$pct02[i] = quantile(var, 0.02, na.rm=TRUE)
      stats$pct05[i] = quantile(var, 0.05, na.rm=TRUE)
      stats$pct25[i] = quantile(var, 0.25, na.rm=TRUE)
      stats$pct50[i] = quantile(var, 0.50, na.rm=TRUE)
      stats$pct75[i] = quantile(var, 0.75, na.rm=TRUE)
      stats$pct95[i] = quantile(var, 0.95, na.rm=TRUE)
      stats$pct98[i] = quantile(var, 0.99, na.rm=TRUE)
      stats$pct99[i] = quantile(var, 0.99, na.rm=TRUE)
      stats$max[i] = max(var, na.rm=TRUE)
      stats$mean[i] = mean(var, na.rm=TRUE)
      stats$std.dev[i] = sd(var, na.rm=TRUE) 
    }
    
    if (!is.numeric(var)) {
      stats$N[i] = length(which(!is.na(var)))
      stats$missing[i] = length(which(is.na(var)))
      stats$missingPct[i] = length(which(is.na(var))) / (length(which(!is.na(var))) + length(which(is.na(var))))
      stats$uniqueVals[i] = length(unique(var))
      stats$corr[i] = NA
      stats$corr_rank[i] = NA
      stats$maxInfCoef[i] = NA
      stats$min[i] = NA
      stats$pct01[i] = NA
      stats$pct02[i] = NA
      stats$pct05[i] = NA
      stats$pct25[i] = NA
      stats$pct50[i] = NA
      stats$pct75[i] = NA
      stats$pct95[i] = NA
      stats$pct98[i] = NA
      stats$pct99[i] = NA
      stats$max[i] = NA
      stats$mean[i] = NA
      stats$std.dev[i] = NA
    }
  }
  
  # If you want to sort variables by what's missing (dplyr package needed)
  if (sortMissing == TRUE) {
    stats = arrange(stats, missing)
  }
  
  # To write to csv
  if (export == TRUE) {
    filename <<- paste('data/summStats_', dataName, '.csv',sep='')
    write.csv(stats, file=filename)
  }
  
  View(stats)
  return(stats)
}
 

# Finds frequency counts for factors (characters and integers with less than 20 unique values)
getFactorFreqs <- function(data, varList, yvar, export=FALSE) {
  # requires dplyr
  dataName <- deparse(substitute(data))
  len <- length(varList)
  
  if (yvar == 'none') {
    data$y <- NA  
  } else {
    data$y <- data[, yvar]
  }
  
  stats <- NULL
  
  for (i in 1:len) {  
    data$var <- data[, varList[i]]
    
    if (is.factor(data$var) | length(unique(data$var)) <= 500) {
      temp <- data %>%
        group_by(var) %>%
        summarise(
          freq = n(),
          YRate = mean(y, na.rm=TRUE)
        ) %>% ungroup()
      
      temp <- arrange(temp, desc(freq))
      rows <- nrow(temp)
      
      stats_add <- data.frame(variable = vector(length=rows),                        
                              level = vector(length=rows),
                              count = vector(length=rows),
                              percent = vector(length=rows),
                              YRate = vector(length=rows))
      
      stats_add$variable <- varList[i]
      stats_add$level <- temp$var
      stats_add$count <- temp$freq
      stats_add$percent <- temp$freq / sum(stats_add$count, na.rm=TRUE)
      stats_add$YRate <- temp$YRate
      
      stats <- rbind(stats, stats_add)
    }
  }
  
  # To write to csv
  if (export == TRUE) {
    filename <<- paste('data/factorFreqs_', dataName, '.csv',sep='')
    write.csv(stats, file=filename)
  }
  
  View(stats)
  return(stats)
}


# Get count and avg responses for factor variables (a.ka. Leave one-out experience variables)
getOneWayVars <- function(train, test, varList, yvar, freq=TRUE, cred=0, rand=0) {
  # freq=TRUE when you want the factor counts; set cred > 0 for credibility adjustment; rand > 0 for random shocking
  # Requires dplyr
  
  len <- length(varList)
  rowNumCheck.train <- nrow(train)
  rowNumCheck.test <- nrow(test)
  
  train$responseVar <- train[, yvar]
  total_avg_response <- mean(train$responseVar, na.rm=TRUE)  
  
  for (i in 1:len) {
    train$groupingVar <- train[, varList[i]]
    test$groupingVar <- test[, varList[i]]   
    
    df <- train %>%
      group_by(groupingVar) %>%
      summarise(
        freq = n() - 1,
        YRate = mean(responseVar, na.rm=TRUE)
      ) %>% ungroup()
    
    train <- left_join(train, df, by='groupingVar')
    
    train_tmp <- unique(train[, c('groupingVar', 'freq', 'YRate')])
    test <- left_join(test, train_tmp, by='groupingVar')
    names(test)[which(names(test)=='freq')] <- 'dummyFreq'
    names(test)[which(names(test)=='YRate')] <- 'dummyRate'
    test$dummyFreq <- test$dummyFreq + 1
    test$dummyFreq[is.na(test$dummyFreq)] <- 0
    
    ids <- which(is.na(test$dummyRate))
    test$dummyRate[ids] <- total_avg_response
    test$dummyRate[-ids] <- (test$dummyRate[-ids] + (total_avg_response * cred / test$dummyFreq[-ids])) * (test$dummyFreq[-ids] / (test$dummyFreq[-ids] + cred))
    
    if (freq) {
      names(test)[which(names(test)=='dummyFreq')] <- paste(varList[i], '_freq', sep='')  
    } else {
      id <- which(names(test)=='dummyFreq')
      test[, id] <- NULL
    }
    
    names(test)[which(names(test)=='dummyRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    # Leave one out adjustment for train data
    train$YRate <- (train$YRate - (train$responseVar / (train$freq+1))) * (train$freq+1)/(train$freq)
    train$YRate <- (train$YRate + (total_avg_response * cred / train$freq)) * (train$freq / (train$freq + cred))
    train$YRate[train$freq == 0] <- total_avg_response
    set.seed(10)
    train$YRate <- train$YRate * (1+(runif(nrow(train))-0.5) * rand)
    
    if (freq) {
      names(train)[which(names(train)=='freq')] <- paste(varList[i], '_freq', sep='')
    } else {
      id <- which(names(train)=='freq')
      train[, id] <- NULL
    }
    
    names(train)[which(names(train)=='YRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    train$groupingVar <- NULL;
    test$groupingVar <- NULL;
  }
  
  train$responseVar <- NULL; train$groupingVar <- NULL; test$groupingVar <- NULL;
  
  if(nrow(train) != rowNumCheck.train) print('Error: Different number of rows in train data. Bad join!')
  
  if(nrow(test) != rowNumCheck.test) print('Error: Different number of rows in test data. Bad join!')
  
  test <<- test
  return(train)
}

# getOneWayVars <- function(train, test, varList, yvar, freq=TRUE, cred=0, rand=0) {
#   # freq=TRUE when you want the factor counts; set cred > 0 for credibility adjustment; rand > 0 for random shocking
#   # Requires dplyr
#   
#   len <- length(varList)
#   rowNumCheck.train <- nrow(train)
#   rowNumCheck.test <- nrow(test)
#   
#   train$responseVar <- train[, yvar]
#   total_avg_response <- mean(train$responseVar, na.rm=TRUE)  
#   
#   for (i in 1:len) {
#     train$groupingVar <- train[, varList[i]]
#     test$groupingVar <- test[, varList[i]]   
#     
#     df <- train %>%
#       group_by(groupingVar) %>%
#       summarise(
#         freq = n() - 1,
#         YRate = mean(responseVar, na.rm=TRUE),
#         YMed = median(responseVar, na.rm=TRUE),
#         YSd = sd(responseVar, na.rm=TRUE)
#       ) %>% ungroup()
#     
#     train <- left_join(train, df, by='groupingVar')
#     
#     train_tmp <- unique(train[, c('groupingVar', 'freq', 'YRate', 'YMed', 'YSd')])
#     test <- left_join(test, train_tmp, by='groupingVar')
#     names(test)[which(names(test)=='freq')] <- 'dummyFreq'
#     names(test)[which(names(test)=='YRate')] <- 'dummyRate'
#     names(test)[which(names(test)=='YMed')] <- 'dummyMed'
#     names(test)[which(names(test)=='YSd')] <- 'dummySd'
#     test$dummyFreq <- test$dummyFreq + 1
#     test$dummyFreq[is.na(test$dummyFreq)] <- 0
#     
#     ids <- which(is.na(test$dummyRate))
#     test$dummyRate[ids] <- total_avg_response
#     test$dummyRate[-ids] <- (test$dummyRate[-ids] + (total_avg_response * cred / test$dummyFreq[-ids])) * (test$dummyFreq[-ids] / (test$dummyFreq[-ids] + cred))
#     
#     if (freq) {
#       names(test)[which(names(test)=='dummyFreq')] <- paste(varList[i], '_freq', sep='')  
#     } else {
#       id <- which(names(test)=='dummyFreq')
#       test[, id] <- NULL
#     }
#     
#     names(test)[which(names(test)=='dummyRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
#     
#     # Leave one out adjustment for train data
#     train$YRate <- (train$YRate - (train$responseVar / (train$freq+1))) * (train$freq+1)/(train$freq)
#     train$YRate <- (train$YRate + (total_avg_response * cred / train$freq)) * (train$freq / (train$freq + cred))
#     train$YRate[train$freq == 0] <- total_avg_response
#     set.seed(10)
#     train$YRate <- train$YRate * (1+(runif(nrow(train))-0.5) * rand)
#     
#     if (freq) {
#       names(train)[which(names(train)=='freq')] <- paste(varList[i], '_freq', sep='')
#     } else {
#       id <- which(names(train)=='freq')
#       train[, id] <- NULL
#     }
#     
#     names(train)[which(names(train)=='YRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
#     
#     train$groupingVar <- NULL;
#     test$groupingVar <- NULL;
#   }
#   
#   train$responseVar <- NULL; train$groupingVar <- NULL; test$groupingVar <- NULL;
#   
#   if(nrow(train) != rowNumCheck.train) print('Error: Different number of rows in train data. Bad join!')
#   
#   if(nrow(test) != rowNumCheck.test) print('Error: Different number of rows in test data. Bad join!')
#   
#   test <<- test
#   return(train)
# }


getOneWayVars_retTrain <- function(train, test, varList, yvar, freq=TRUE, cred=0, rand=0) {
  # freq=TRUE when you want the factor counts; set cred > 0 for credibility adjustment; rand > 0 for random shocking
  # Requires dplyr
  
  len <- length(varList)
  rowNumCheck.train <- nrow(train)
  rowNumCheck.test <- nrow(test)
  
  train$responseVar <- train[, yvar]
  total_avg_response <- mean(train$responseVar, na.rm=TRUE) 
  
  for (i in 1:len) {
    train$groupingVar <- train[, varList[i]]
    test$groupingVar <- test[, varList[i]]   
    
    df <- train %>%
      group_by(groupingVar) %>%
      summarise(
        freq = n() - 1,
        YRate = mean(responseVar, na.rm=TRUE)
      ) %>% ungroup()
    
    train <- left_join(train, df, by='groupingVar')
    
    train_tmp <- unique(train[, c('groupingVar', 'freq', 'YRate')])
    test <- left_join(test, train_tmp, by='groupingVar')
    names(test)[which(names(test)=='freq')] <- 'dummyFreq'
    names(test)[which(names(test)=='YRate')] <- 'dummyRate'
    test$dummyFreq <- test$dummyFreq + 1
    test$dummyFreq[is.na(test$dummyFreq)] <- 0
    
    ids <- which(is.na(test$dummyRate))
    test$dummyRate[ids] <- total_avg_response
    test$dummyRate[-ids] <- (test$dummyRate[-ids] + (total_avg_response * cred / test$dummyFreq[-ids])) * (test$dummyFreq[-ids] / (test$dummyFreq[-ids] + cred))
    
    if (freq) {
      names(test)[which(names(test)=='dummyFreq')] <- paste(varList[i], '_freq', sep='')  
    } else {
      id <- which(names(test)=='dummyFreq')
      test[, id] <- NULL
    }
    
    names(test)[which(names(test)=='dummyRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    # Leave one out adjustment for train data
    train$YRate <- (train$YRate - (train$responseVar / (train$freq+1))) * (train$freq+1)/(train$freq)
    train$YRate <- (train$YRate + (total_avg_response * cred / train$freq)) * (train$freq / (train$freq + cred))
    train$YRate[train$freq == 0] <- total_avg_response
    set.seed(10)
    train$YRate <- train$YRate * (1+(runif(nrow(train))-0.5) * rand)
    
    if (freq) {
      names(train)[which(names(train)=='freq')] <- paste(varList[i], '_freq', sep='')
    } else {
      id <- which(names(train)=='freq')
      train[, id] <- NULL
    }
    
    names(train)[which(names(train)=='YRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    train$groupingVar <- NULL;
    test$groupingVar <- NULL;
  }
  
  train$responseVar <- NULL; train$groupingVar <- NULL; test$groupingVar <- NULL;
  
  if(nrow(train) != rowNumCheck.train) print('Error: Different number of rows in train data. Bad join!')
  
  if(nrow(test) != rowNumCheck.test) print('Error: Different number of rows in test data. Bad join!')
  

  return(train)
}

getOneWayVars_retTest <- function(train, test, varList, yvar, freq=TRUE, cred=0, rand=0) {
  # freq=TRUE when you want the factor counts; set cred > 0 for credibility adjustment; rand > 0 for random shocking
  # Requires dplyr
  
  len <- length(varList)
  rowNumCheck.train <- nrow(train)
  rowNumCheck.test <- nrow(test)
  
  train$responseVar <- train[, yvar]
  total_avg_response <- mean(train$responseVar, na.rm=TRUE) 
  
  for (i in 1:len) {
    train$groupingVar <- train[, varList[i]]
    test$groupingVar <- test[, varList[i]]   
    
    df <- train %>%
      group_by(groupingVar) %>%
      summarise(
        freq = n() - 1,
        YRate = mean(responseVar, na.rm=TRUE)
      ) %>% ungroup()
    
    train <- left_join(train, df, by='groupingVar')
    
    train_tmp <- unique(train[, c('groupingVar', 'freq', 'YRate')])
    test <- left_join(test, train_tmp, by='groupingVar')
    names(test)[which(names(test)=='freq')] <- 'dummyFreq'
    names(test)[which(names(test)=='YRate')] <- 'dummyRate'
    test$dummyFreq <- test$dummyFreq + 1
    test$dummyFreq[is.na(test$dummyFreq)] <- 0
    
    ids <- which(is.na(test$dummyRate))
    test$dummyRate[ids] <- total_avg_response
    test$dummyRate[-ids] <- (test$dummyRate[-ids] + (total_avg_response * cred / test$dummyFreq[-ids])) * (test$dummyFreq[-ids] / (test$dummyFreq[-ids] + cred))
    
    if (freq) {
      names(test)[which(names(test)=='dummyFreq')] <- paste(varList[i], '_freq', sep='')  
    } else {
      id <- which(names(test)=='dummyFreq')
      test[, id] <- NULL
    }
    
    names(test)[which(names(test)=='dummyRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    # Leave one out adjustment for train data
    train$YRate <- (train$YRate - (train$responseVar / (train$freq+1))) * (train$freq+1)/(train$freq)
    train$YRate <- (train$YRate + (total_avg_response * cred / train$freq)) * (train$freq / (train$freq + cred))
    train$YRate[train$freq == 0] <- total_avg_response
    set.seed(10)
    train$YRate <- train$YRate * (1+(runif(nrow(train))-0.5) * rand)
    
    if (freq) {
      names(train)[which(names(train)=='freq')] <- paste(varList[i], '_freq', sep='')
    } else {
      id <- which(names(train)=='freq')
      train[, id] <- NULL
    }
    
    names(train)[which(names(train)=='YRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    train$groupingVar <- NULL;
    test$groupingVar <- NULL;
  }
  
  train$responseVar <- NULL; train$groupingVar <- NULL; test$groupingVar <- NULL;
  
  if(nrow(train) != rowNumCheck.train) print('Error: Different number of rows in train data. Bad join!')
  
  if(nrow(test) != rowNumCheck.test) print('Error: Different number of rows in test data. Bad join!')
  
  return(test)
}


# Changes class type of variables --------------------------------------------------------
makeFactor <- function(data, varList) {
  for (i in varList) {
    data[, i] <- as.factor(data[, i])
  }
  
  return(data)
}


percentile <- function(x) rank(x, na.last='keep')/length(which(!is.na(x)))

binVariable <- function(x, bins) {
  # requires dplyr package
  ntile(x, bins)
}

imputeWithMean <- function(data, vars) {
  for (i in vars) {
    data[is.na(data[, i]), i] <- mean(as.numeric(data[, i]), na.rm=TRUE)
  }
    
  return(data)
}


# For response vs. x-variable decile plots (can only do one variable at a time) ------------------------------
getUnivariatePlots <- function(data, bins=10, xvar, yvar, facet=NULL) {
  # Requires ggplot2, Hmisc, dplyr, lubridate
  
  data$x <- data[, xvar]
  data$y <- data[, yvar]
  
  if (!is.null(facet)) {
    data$facetVar <- data[, facet]
    
    df <- data %>%
      group_by(facetVar) %>%
      mutate(bin = as.numeric(cut2(x, g=bins)))
    
    df <- df %>%
      group_by(facetVar, bin) %>%
      summarise(avg = mean(y, na.rm=TRUE))
    
    ggplot(df, aes(x=bin, y=avg)) +
      geom_bar(stat='identity', fill='dodgerblue4', colour='black') +
      ylab("Response Rate\n") +
      ggtitle(xvar) +
      xlab('\nBins') +
      facet_wrap( ~ facetVar) +
      scale_x_continuous(breaks=1:bins) +
      theme(plot.title = element_text(size=20)) +
      theme(axis.title.x = element_text(size=20)) +
      theme(axis.title.y = element_text(size=20)) +
      theme(axis.text.x = element_text(size=16)) + 
      theme(axis.text.y = element_text(size=16)) +
      theme(strip.text = element_text(size=16))
  } else if (is.null(facet)) {  
    
    data$bin <- as.numeric(cut2(data$x, g=bins))
    df <- data %>%
      group_by(bin) %>%
      summarise(avg = mean(y, na.rm=TRUE))
    
    ggplot(df, aes(x=bin, y=avg)) +
      geom_bar(stat='identity', fill='dodgerblue4', colour='black') +
      ylab("Response Rate\n") +
      ggtitle(xvar) +
      xlab('\nBins') +
      scale_x_continuous(breaks=1:bins) +
      theme(plot.title = element_text(size=20)) +
      theme(axis.title.x = element_text(size=20)) +
      theme(axis.title.y = element_text(size=20)) +
      theme(axis.text.x = element_text(size=16)) + 
      theme(axis.text.y = element_text(size=16)) +
      theme(strip.text = element_text(size=16))
  }
}


# Creates histograms of variables ------------------------------
plotHistogram <- function(data, varList) {
  # Requires ggplot2
  
  for (var in varList) {
    data$var <- data[, var]  
    
    if (is.numeric(data$var)) {  
      # Full distribution
      hist.obj <- hist(data$var, plot=FALSE)
      hist.df <- data.frame(mids=hist.obj$mids, density=hist.obj$counts/sum(hist.obj$counts))  
      
      graph.hist.full <- ggplot(hist.df, aes(x=mids, y=density * 100)) +
        geom_bar(stat='identity', fill='steel blue', colour='black') +
        ggtitle('Full Distribution\n') +
        scale_x_continuous(breaks=hist.obj$breaks) +
        scale_y_continuous('Probability (%)\n') +
        xlab(var)
      
      # 0-25 percentile distribution
      x <- data$var
      x <- subset(x, x < quantile(x, 0.25, na.rm=TRUE))
      hist.obj.left25 <- hist(x, plot=FALSE)
      hist.df.left25 <- data.frame(mids=hist.obj.left25$mids, counts=hist.obj.left25$counts)
      graph.hist.left25 <- ggplot(hist.df.left25, aes(x=mids, y=counts)) +        
        geom_bar(stat='identity', fill='steel blue', colour='black') +
        ggtitle('0 - 25 percentiles') +
        scale_x_continuous(breaks=hist.obj.left25$breaks, limits=c(min(hist.obj.left25$breaks), max(hist.obj.left25$breaks))) +
        scale_y_continuous("Count\n") +
        xlab(var)
      
      # 25-75 percentile distribution
      x <- data$var
      x <- subset(x, quantile(x, 0.25, na.rm=TRUE) < x & x < quantile(x, 0.75, na.rm=TRUE))
      hist.obj.mid50 <- hist(x, plot=FALSE)
      hist.df.mid50 <- data.frame(mids=hist.obj.mid50$mids, counts=hist.obj.mid50$counts)
      
      graph.hist.mid50 <- ggplot(hist.df.mid50, aes(x=mids, y=counts)) +  
        geom_bar(stat='identity', fill='steel blue', colour='black') +
        ggtitle('25 - 75 percentiles') +
        scale_x_continuous(breaks=hist.obj.mid50$breaks, limits=c(min(hist.obj.mid50$breaks), max(hist.obj.mid50$breaks))) +
        scale_y_continuous("Count\n") +
        xlab(var)
      
      # 75-100 percentile distribution
      x <- data$var
      x <- subset(x, x > quantile(x, 0.75, na.rm=TRUE))
      hist.obj.right25 <- hist(x, plot=FALSE)
      hist.df.right25 <- data.frame(mids=hist.obj.right25$mids, counts=hist.obj.right25$counts)
      
      graph.hist.right25 <- ggplot(hist.df.right25, aes(x=mids, y=counts)) +
        geom_bar(stat='identity', fill='steel blue', colour='black') +
        ggtitle('75 - 100 percentiles') +
        scale_x_continuous(breaks=hist.obj.right25$breaks, limits=c(min(hist.obj.right25$breaks), max(hist.obj.right25$breaks))) +
        scale_y_continuous("Count\n") +  
        xlab(var)
      
      multiplot(graph.hist.full, graph.hist.left25, graph.hist.mid50, graph.hist.right25, cols=2)     
    } 
  }  
  
}


# Calculates percentile ------------------------------
percentile <- function(x) rank(x, na.last='keep')/length(which(!is.na(x)))

# Changes class type of variables ------------------------------
makeFactor <- function(data, varList) {
  for (i in varList) {
    data[, i] <- as.factor(data[, i])
  }
  
  return(data)
}

makeNumeric <- function(data, varList) {
  for (i in varList) {
    data[, i] <- as.numeric(data[, i])
  }
  
  return(data)
}


# Produces partial dependence plots for gbm models ------------------------------
getPDplots_scaled <- function(gbmObject, varList, n.trees, trainingData) {
  
  require(gbm); require(ggplot2); require(scales);
  trainingData = as.data.frame(trainingData);
  
  for (var in varList) {  
    if (is.numeric(trainingData[, var])) {
      pd.df <<- plot.gbm(gbmObject, var, n.trees, return.grid=TRUE);
      graph.pd <- ggplot(pd.df, aes(x=pd.df[, 1], y=exp(y)/(exp(y) + 1) * 100)) +
        geom_line(col='dodgerblue4', lwd=1.05, alpha=0.8) +
        ylab("Probability (%)\n") +
        xlab(var) +
        scale_x_continuous(labels = comma) +
        theme(axis.title.x = element_text(size=20)) +
        theme(axis.title.y = element_text(size=20, angle=90)) +
        theme(axis.text.x = element_text(size=16)) + 
        theme(axis.text.y = element_text(size=16));
      
      hist.obj <- hist(trainingData[, var], plot=FALSE);
      
      hist.df <<- data.frame(mids=hist.obj$mids, density=hist.obj$counts/sum(hist.obj$counts))
      
      graph.hist <- ggplot(hist.df, aes(x=mids, y=density * 100)) +
        geom_bar(stat='identity', alpha=0.25, fill='darkred') +
        scale_y_continuous("%\n") +
        xlab(var) +
        scale_x_continuous(labels = comma) +
        theme(axis.title.x = element_text(size=20)) +
        theme(axis.title.y = element_text(size=20, angle=90)) +
        theme(axis.text.x = element_text(size=16)) + 
        theme(axis.text.y = element_text(size=16));
      
      multiplot(graph.pd, graph.hist);
      
      next;      
    }
    
    if (is.factor(trainingData[, var])) {
      
      pd.df <<- plot.gbm(gbmObject, var, n.trees, return.grid=TRUE);
      
      graph.pd <- ggplot(pd.df, aes(x=pd.df[, 1], y=exp(y)/(exp(y) + 1) * 100)) +
        geom_point(pch=15, col='dodgerblue4', size=3, alpha=0.8) +
        ylab("Probability (%)\n") +
        xlab(var) +
        theme(axis.text.x = element_text(angle=350)) +
        theme(axis.title.x = element_text(size=20)) +
        theme(axis.title.y = element_text(size=20, angle=90)) +
        theme(axis.text.x = element_text(size=16)) + 
        theme(axis.text.y = element_text(size=16));
      
      hist_df <<- count(trainingData, var);
      
      hist_df$density <<- hist_df$freq / sum(hist_df$freq);
      
      graph.hist <- ggplot(hist_df, aes(x=hist_df[, 1], y=density * 100)) +
        geom_bar(stat='identity', alpha=0.25, fill='darkred') +
        scale_y_continuous("%\n") +
        xlab(var) +
        theme(axis.text.x = element_text(angle=350)) +
        theme(axis.title.x = element_text(size=20)) +
        theme(axis.title.y = element_text(size=20, angle=90)) +
        theme(axis.text.x = element_text(size=16)) +   
        theme(axis.text.y = element_text(size=16));
      
      multiplot(graph.pd, graph.hist);   
    } 
  }
}


getPDplots_scaled_log <- function(gbmObject, varList, n.trees, trainingData) {
  
  require(gbm); require(ggplot2); require(scales);
  
  trainingData = as.data.frame(trainingData)
  
  for (var in varList) {  
    
    if (is.numeric(trainingData[, var])) {
      pd.df <<- plot.gbm(gbmObject, var, n.trees, return.grid=TRUE)
      graph.pd <- ggplot(pd.df, aes(x=pd.df[, 1], y=exp(y)/(exp(y) + 1) * 100)) +
        geom_line(col='dodgerblue4', lwd=1.05, alpha=0.8) +
        ylab("Probability (%)\n") +
        xlab(var) +
        scale_x_log10(labels = comma) +
        theme(axis.title.x = element_text(size=20)) +
        theme(axis.title.y = element_text(size=20, angle=90)) +
        theme(axis.text.x = element_text(size=16)) + 
        theme(axis.text.y = element_text(size=16))
      
      hist.obj <- hist(trainingData[, var], plot=FALSE);
      hist.df <<- data.frame(mids=hist.obj$mids, density=hist.obj$counts/sum(hist.obj$counts))
      
      graph.hist <- ggplot(hist.df, aes(x=mids, y=density * 100)) +  
        geom_bar(stat='identity', alpha=0.25, fill='darkred') +
        scale_y_continuous("%\n") +
        xlab(var) +
        scale_x_log10(labels = comma) +
        theme(axis.title.x = element_text(size=20)) +   
        theme(axis.title.y = element_text(size=20, angle=90)) +  
        theme(axis.text.x = element_text(size=16)) +      
        theme(axis.text.y = element_text(size=16));
      
      multiplot(graph.pd, graph.hist); 
      next;  
    }
    
    if (is.factor(trainingData[, var])) {    
      pd.df <<- plot.gbm(gbmObject, var, n.trees, return.grid=TRUE); 
      graph.pd <- ggplot(pd.df, aes(x=pd.df[, 1], y=exp(y)/(exp(y) + 1) * 100)) +
        geom_point(pch=15, col='dodgerblue4', size=3, alpha=0.8) +     
        ylab("Probability (%)\n") +
        xlab(var) +
        theme(axis.text.x = element_text(angle=350)) +
        theme(axis.title.x = element_text(size=20)) +
        theme(axis.title.y = element_text(size=20, angle=90)) +
        theme(axis.text.x = element_text(size=16)) + 
        theme(axis.text.y = element_text(size=16))
      
      hist_df <<- count(trainingData, var)
      
      hist_df$density <<- hist_df$freq / sum(hist_df$freq)
      
      graph.hist <- ggplot(hist_df, aes(x=hist_df[, 1], y=density * 100)) +       
        geom_bar(stat='identity', alpha=0.25, fill='darkred') +
        scale_y_continuous("%\n") +
        xlab(var) +
        theme(axis.text.x = element_text(angle=350)) +
        theme(axis.title.x = element_text(size=20)) +
        theme(axis.title.y = element_text(size=20, angle=90)) +
        theme(axis.text.x = element_text(size=16)) + 
        theme(axis.text.y = element_text(size=16));
      
      multiplot(graph.pd, graph.hist)  
    } 
  }
}


# Puts multiple ggplots on same page ------------------------------
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols)) 
  }
  
  if (numPlots==1) {
    print(plots[[1]])  
  } else {
    # Set up the page
    grid.newpage() 
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,                              
                                      layout.pos.col = matchidx$col))     
    }  
  }
}