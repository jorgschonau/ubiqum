rm(list = ls())
#### Libraries ####
library(readr)
library(caret)
library(e1071)
library(reshape2)
library(tidyverse)
library(tictoc)
library(plotly)

setwd("~/Desktop/Ubiqum/Ubiqum mentor/Task 8 Wifi/")

#### Data import, type conversions, basic pruning ####
trainingData <- as.data.frame(read_csv(file="trainingData.csv",
                                       col_types = cols(BUILDINGID = col_character(), 
                                       FLOOR = col_character(), PHONEID = col_character(), 
                                       RELATIVEPOSITION = col_character(), 
                                       USERID = col_character())))

validationData <- as.data.frame(read_csv(file="validationData.csv",
                                         col_types = cols(BUILDINGID = col_character(), 
                                                          FLOOR = col_character(), PHONEID = col_character(), 
                                                          RELATIVEPOSITION = col_character(), 
                                                          USERID = col_character())))

# adding train/val tags to each partition
trainingData$Partition <- "train" 
validationData$Partition <- "val" 
# merging training/validation
trainingData <- rbind(trainingData,validationData)

# --- data type conversion etc 
trainingData$FLOOR = as.factor(trainingData$FLOOR)
trainingData$LATITUDE = as.integer(trainingData$LATITUDE)
trainingData$LONGITUDE = as.integer(trainingData$LONGITUDE)
trainingData$BUILDINGID = as.factor(trainingData$BUILDINGID)
trainingData$SPACEID = as.factor(trainingData$SPACEID)
trainingData$RELATIVEPOSITION = as.factor(trainingData$RELATIVEPOSITION)
# removing useless variables (we need user id & device id for histograms, 
# can delete later, will also keep timestamp this time)
# trainingData$USERID <- NULL
# trainingData$PHONEID <- NULL
trainingData$TIMESTAMP <- NULL

# adding uniqe space id - might be useful for later 
trainingData$UNIQUESPACEID <- as.factor(paste(trainingData$BUILDINGID, trainingData$FLOOR, trainingData$SPACEID, trainingData$RELATIVEPOSITION, sep=""))
# adding building/ floor column, might be useful for later
trainingData$BUILDFLOOR <- as.factor(paste(trainingData$BUILDINGID,
                                           trainingData$FLOOR,
                                           sep="-"))

# adding ID no - might be useful for later (joining etc)
trainingData$ID <- seq.int(nrow(trainingData))

# #### Some data exploration ####
# # df size
# print(object.size(trainingData),units="Mb")
# apply(trainingData== 100, 2, sd)
# # cols with sd/ car smaller then
# trainingData[, which(apply(trainingData[,1:(ncol(trainingData)-12)],2,sd) < 2)]
# cols with unique values less than ... 
# trainingData[,1:(ncol(trainingData)-12)][, -which(length(unique(trainingData[,1:(ncol(trainingData)-12)])) > 10)])
# trainingData[,1:(ncol(trainingData)-12)][, -which(length(unique(trainingData[,1:(ncol(trainingData)-12)])) > 10)]
# trainingData[,1:(ncol(trainingData)-12)][, -which(length(unique(trainingData[,1:(ncol(trainingData)-12)])) > 10)]
# which(length(unique(trainingData[,1:(ncol(trainingData)-12)])) > 20)
# length(apply(trainingData[,1:(ncol(trainingData)-12)],2,unique)) <20

#### removing waps with zero variance 

# max(trainingData[1:(ncol(trainingData)-13)])
# min(trainingData[1:(ncol(trainingData)-13)])

# filtering out low signals, replacing  <=-95 by -100
for(i in 1:(ncol(trainingData)-12)){
  trainingData[which(trainingData[,i] < -92), i] = -100
}

# replacing 100 by -100 for normilisation
for(i in 1:(ncol(trainingData)-12)){
  trainingData[which(trainingData[,i] == 100), i] = -100
}

trainingData <- trainingData[-which(apply(trainingData[,1:(ncol(trainingData)-12)], 2,
                                      var) == 0)] # 38 waps, so could have used the step above

#ncol(trainingData) 475 cols
#which(vapply(trainingData[, 1:(ncol(trainingData) - 12)], function(x) length(unique(x)) <= 5 & max(x) <= -80, logical(1L)) =="TRUE")

trainingData_small <- trainingData[-which(vapply(trainingData[, 1:(ncol(trainingData) - 12)],
                                                 function(x) var(x) <= 4 & max(x) <= -74, 
                                                 logical(1L)) =="TRUE")]

# ncol(trainingData_small) 333 cols

#trainingData_small <- trainingData[-which(vapply(trainingData[, 1:(ncol(trainingData) - 12)],
#                                                 function(x) var(x) <= 2 & max(x) <= -75, logical(1L)) =="TRUE")]

ncol(trainingData)
ncol(trainingData_small)
# from now working with two df's: trainingData & trainingData_small, still need to split them in 
# training and val again


# # Hist of wap signal strength distribution 
# 
# training_melt <- melt(trainingData[,1:(ncol(trainingData)-12)])
# training_melt <- subset(training_melt, value != -100 )
# # 
# ggplot(training_melt, aes(value)) + geom_histogram(binwidth = 1) 

# training_melt_small <- melt(trainingData_small[,1:(ncol(trainingData_small)-12)])
# training_melt_small <- subset(training_melt_small, value != -100 )
# # 
# ggplot(training_melt_small, aes(value)) + geom_histogram(binwidth = 1) 

# adding column with max value of each row
trainingData$max <- apply(trainingData[,1:(ncol(trainingData)-12)], 1, FUN=max)
trainingData_small$max <- apply(trainingData_small[,1:(ncol(trainingData_small)-12)], 1, FUN=max)

# # Hist of wap signal strength distribution 
# 
# training_melt <- melt(trainingData[,1:(ncol(trainingData)-13)])
# training_melt <- subset(training_melt, value != -100 )
# # 
# ggplot(training_melt, aes(value)) + geom_histogram(binwidth = 1) 
# 
# training_melt_small <- melt(trainingData_small[,1:(ncol(trainingData_small)-13)])
# training_melt_small <- subset(training_melt_small, value != -100 )
# # 
# ggplot(training_melt_small, aes(value)) + geom_histogram(binwidth = 1) 

# deleting rows with user #6

trainingData_small <- trainingData_small[-which(trainingData_small$USERID ==6) ,]
trainingData <- trainingData[-which(trainingData$USERID ==6),]

# deleting all other rows with row max > -30
trainingData_small <- trainingData_small[-which(trainingData_small$max >-30),]
trainingData <- trainingData[-which(trainingData$max >-30),]


#### Reverse values (into positives) ####

trainingData[,1:(ncol(trainingData)-13)] <- apply(trainingData[,1:(ncol(trainingData)-13)], 2,
                                        function(x) 100 + x)
trainingData_small[,1:(ncol(trainingData_small)-13)] <- apply(trainingData_small[,1:(ncol(trainingData_small)-13)], 2,
                                                  function(x) 100 + x)

# trainingData[,1:(ncol(trainingData)-13)] <- apply(trainingData[,1:(ncol(trainingData)-13)], 2,
#                                                   function(x) 100 + x)

# updating max values per row
trainingData$max <- apply(trainingData[,1:(ncol(trainingData)-12)], 1, FUN=max)
trainingData_small$max <- apply(trainingData_small[,1:(ncol(trainingData_small)-12)], 1, FUN=max)

# removing row with max = 0 (helps to avoid nans in normalisation by row)
trainingData_small <- trainingData_small[-which(trainingData_small$max  == 0),]
trainingData <- trainingData[-which(trainingData$max == 0),]


# ----> Normalisation (for real) ----
# check that everything is right
max(trainingData[,1:(ncol(trainingData)-13)])
min(trainingData[,1:(ncol(trainingData)-13)])

max(trainingData_small[,1:(ncol(trainingData_small)-13)])
min(trainingData_small[,1:(ncol(trainingData_small)-13)])

normalize <- function(x) {
return (x / 70 ) #global max is 70
} # max(training_less_norm)

normalizebyrow <- function(x) {
  return ( (x-min(x))/(max(x)-min(x)) )
} # max(training_less_norm)

#remove max 

trainingData[,1:(ncol(trainingData)-13)] <- round(apply(trainingData[,1:(ncol(trainingData)-13)], 2,
                                                 normalize), 6)
#trainingData_small[,1:(ncol(trainingData_small)-13)] <- round(apply(trainingData_small[,1:(ncol(trainingData_small)-13)], 2,
#                                                        normalize), 6)

# option B: normalize by row
#training_norm <- as.data.frame(t(apply(trainingData_small,1,normalizebyrow)))

trainingData_small[,1:(ncol(trainingData_small)-13)] <- round(t(apply(trainingData_small[,1:(ncol(trainingData_small)-13)],
                                                        1, normalizebyrow)), 6)

# normalisation step2: +1, log & division by log(2(max value)) to set scale 0 - 1
# training_norm <- training_norm + 1
# training_norm <- log(training_norm)
# training_norm <- training_norm/log(2)

# splitting back into train/ val
training_big <- trainingData %>% dplyr::filter(Partition =="train")
validation_big <- trainingData %>% dplyr::filter(Partition =="val")

training_small <- trainingData_small %>% dplyr::filter(Partition =="train")
validation_small <- trainingData_small %>% dplyr::filter(Partition =="val")

#### Modelling! ####

#---- Create Training/Testing Partitions ----- 
set.seed(123)
sampleIndex <- sample(1:nrow(training_big), 3000) #<---- anything higher much higher than 2000 will slow down training off the models considerably 
sampleIndex2 <- sample(1:nrow(training_small), 3000)

# create Sample Dataframes for trainining_big
training_big_sample <- training_big[sampleIndex,]

inTraining <- createDataPartition(training_big_sample$BUILDINGID, p = .80, list = FALSE)
training_big_training <- na.omit(training_big_sample[inTraining,])
testing_big_training <- training_big_sample[-inTraining,]

# create Sample Dataframes for trainining_small
training_small_sample <- training_small[sampleIndex2,]

inTraining <- createDataPartition(training_small_sample$BUILDINGID, p = .75, list = FALSE)
training_small_training <- na.omit(training_small_sample[inTraining,])
testing_small_training <- training_small_sample[-inTraining,]

##### ---- STEP 1: Predicting Building ID ####
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# ---- > SVM Building ----

# model_svm_bld_big <- train(BUILDINGID ~ ., data = training_big_training[, c(1:(ncol(training_big_training)-13), 
#                                                                             (ncol(training_big_training)-9))],
#                      method = "svmLinear2", 
#                      trControl = fitControl)

model_svm_bld_small <- train(BUILDINGID ~ .,
                             data = training_small_training[, c(1:(ncol(training_small_training)-13), 
                                                                (ncol(training_small_training)-9))],
                           method = "svmLinear2", 
                           trControl = fitControl)

# svm_pred_build_big <- predict(model_svm_bld_big, testing_big_training)
# postResample(svm_pred_build_big, testing_big_training$BUILDINGID)

svm_pred_build_small <- predict(model_svm_bld_small, testing_small_training)
postResample(svm_pred_build_small, testing_small_training$BUILDINGID)


# ---- > knn Building ----

# model_knn_bld_big <- train(BUILDINGID ~ ., data = training_big_training[, c(1:(ncol(training_big_training)-13), 
#                                                                             (ncol(training_big_training)-9))],
#                            method = "knn", 
#                            preProcess = c("center","scale"),
#                            trControl = fitControl)

model_knn_bld_small <- train(BUILDINGID ~ ., data = training_small_training[, c(1:(ncol(training_small_training)-13), 
                                                                                (ncol(training_small_training)-9))],
                             method = "knn",
                             preProcess = c("center","scale"),
                             trControl = fitControl)

# knn_pred_build_big <- predict(model_knn_bld_big, testing_big_training)
# postResample(knn_pred_build_big, testing_big_training$BUILDINGID)

knn_pred_build_small <- predict(model_knn_bld_small, testing_small_training)
postResample(knn_pred_build_small, testing_small_training$BUILDINGID)


# ----2. Step Floor ----

# ---- > SVM floor ----

# model_svm_floor_big <- train(FLOOR ~ ., data = training_big_training[, c(1:(ncol(training_big_training)-13), 
#                                                                             (ncol(training_big_training)-10))],
#                            method = "svmLinear2", 
#                            trControl = fitControl)

model_svm_floor_small <- train(FLOOR ~ .,
                             data = training_small_training[, c(1:(ncol(training_small_training)-13), 
                                                                (ncol(training_small_training)-10))],
                             method = "svmLinear2", 
                             trControl = fitControl)

# svm_pred_floor_big <- predict(model_svm_floor_big, testing_big_training)
# postResample(svm_pred_floor_big, testing_big_training$FLOOR)

svm_pred_floor_small <- predict(model_svm_floor_small, testing_small_training)
postResample(svm_pred_floor_small, testing_small_training$FLOOR)

# ---- > knn floor----
# 
# model_knn_floor_big <- train(FLOOR ~ ., data = training_big_training[, c(1:(ncol(training_big_training)-13), 
#                                                                             (ncol(training_big_training)-10))],
#                            method = "knn", 
#                            preProcess = c("center","scale"),
#                            trControl = fitControl)

model_knn_floor_small <- train(FLOOR ~ ., data = training_small_training[, c(1:(ncol(training_small_training)-13), 
                                                                                (ncol(training_small_training)-10))],
                             method = "knn",
                             preProcess = c("center","scale"),
                             trControl = fitControl)

# knn_pred_floor_big <- predict(model_knn_floor_big, testing_big_training)
# postResample(knn_pred_floor_big, testing_big_training$FLOOR)

knn_pred_floor_small <- predict(model_knn_floor_small, testing_small_training)
postResample(knn_pred_floor_small, testing_small_training$FLOOR)

# ---- 3rd step coords
# ---- 3rd step A: Latitude
# lat

# model_svm_latitude_big <- train(LATITUDE ~ ., data = training_big_training[, c(1:(ncol(training_big_training)-13), 
#                                                                          (ncol(training_big_training)-11))],
#                              method = "svmLinear2", 
#                              trControl = fitControl)
# 
# model_svm_latitude_small <- train(LATITUDE ~ .,
#                                data = training_small_training[, c(1:(ncol(training_small_training)-13), 
#                                                                   (ncol(training_small_training)-11))],
#                                method = "svmLinear2", 
#                                trControl = fitControl)
# 
# svm_pred_latitude_big <- predict(model_svm_latitude_big, testing_big_training)
# postResample(svm_pred_latitude_big, testing_big_training$LATITUDE)
# 
# svm_pred_latitude_small <- predict(model_svm_latitude_small, testing_small_training)
# postResample(svm_pred_latitude_small, testing_small_training$LATITUDE)

# ---- > knn latitude----

# model_knn_latitude_big <- train(LATITUDE ~ ., data = training_big_training[, c(1:(ncol(training_big_training)-13), 
#                                                                          (ncol(training_big_training)-11))],
#                              method = "knn", 
#                              preProcess = c("center","scale"),
#                              trControl = fitControl)

model_knn_latitude_small <- train(LATITUDE ~ ., data = training_small_training[, c(1:(ncol(training_small_training)-13), 
                                                                             (ncol(training_small_training)-11))],
                               method = "knn",
                               preProcess = c("center","scale"),
                               trControl = fitControl)

# knn_pred_latitude_big <- predict(model_knn_latitude_big, testing_big_training)
# postResample(knn_pred_latitude_big, testing_big_training$LATITUDE)

knn_pred_latitude_small <- predict(model_knn_latitude_small, testing_small_training)
postResample(knn_pred_latitude_small, testing_small_training$LATITUDE)

# 3rd step b: longitude
# longitude

# model_svm_longitude_big <- train(LONGITUDE ~ ., data = training_big_training[, c(1:(ncol(training_big_training)-13), 
#                                                                                (ncol(training_big_training)-12))],
#                                 method = "svmLinear2", 
#                                 trControl = fitControl)
# 
# model_svm_longitude_small <- train(LONGITUDE ~ .,
#                                   data = training_small_training[, c(1:(ncol(training_small_training)-13), 
#                                                                      (ncol(training_small_training)-12))],
#                                   method = "svmLinear2", 
#                                   trControl = fitControl)
# 
# svm_pred_longitude_big <- predict(model_svm_longitude_big, testing_big_training)
# postResample(svm_pred_longitude_big, testing_big_training$LONGITUDE)
# 
# svm_pred_longitude_small <- predict(model_svm_longitude_small, testing_small_training)
# postResample(svm_pred_longitude_small, testing_small_training$LONGITUDE)

# ---- > knn longitude ----

# model_knn_longitude_big <- train(LONGITUDE ~ ., data = training_big_training[, c(1:(ncol(training_big_training)-13), 
#                                                                                (ncol(training_big_training)-12))],
#                                 method = "knn", 
#                                 preProcess = c("center","scale"),
#                                 trControl = fitControl)

model_knn_longitude_small <- train(LONGITUDE ~ ., data = training_small_training[, c(1:(ncol(training_small_training)-13), 
                                                                                   (ncol(training_small_training)-12))],
                                  method = "knn", 
                                  preProcess = c("center","scale"),
                                  trControl = fitControl)

# knn_pred_longitude_big <- predict(model_knn_longitude_big, testing_big_training)
# postResample(knn_pred_longitude_big, testing_big_training$LONGITUDE)

knn_pred_longitude_small <- predict(model_knn_longitude_small, testing_small_training)
postResample(knn_pred_longitude_small, testing_small_training$LONGITUDE)

length(knn_pred_longitude_small)
nrow(testing_small_training)
length(testing_small_training$LONGITUDE)

View(testing_small_training)

#### Best Models ####

model_svm_bld_small
model_svm_floor_small
model_knn_latitude_small
knn_pred_longitude_small

#### predictions ####

pred_build_val <- predict(model_svm_bld_small, validation_small)
postResample(pred_build_val,validation_small$BUILDINGID)
validation_small$predBUILD <- pred_build_val

pred_floor_val <- predict(model_svm_floor_small, validation_small)
postResample(pred_floor_val,validation_small$FLOOR)
validation_small$predFLOOR <- pred_floor_val

pred_long_val <- predict(model_knn_longitude_small, validation_small)
postResample(pred_long_val,validation_small$LONGITUDE)
validation_small$predLONG <- pred_long_val

pred_lat_val <- predict(model_knn_latitude_small, validation_small)
postResample (pred_lat_val,validation_small$LATITUDE)
validation_small$predlat <- pred_lat_val

# error distance
validation_small$error_distance <- sqrt ( (validation_small$LATITUDE - validation_small$predlat)^2 + (validation_small$LONGITUDE - validation_small$predLONG)^2 )
summary(validation_small$error_distance)
boxplot(validation_small$error_distance)

View(validation_small[which(validation_small$error_distance>150),])


validation_small$predBUILDFLOOR <- as.factor(paste(validation_small$predBUILD,
                                           validation_small$predFLOOR,
                                           sep="-"))


View(validation_small[which(validation_small$BUILDFLOOR != validation_small$predBUILDFLOOR),])


plot(training_big[ncol(:ncol(training_big)])
                  
View(head(training_big,5))

# RMSE  Rsquared       MAE 
# 22.481079  0.965351  9.201476 
# > postResample (pred_lat_val,validation_small$LATITUDE)
# RMSE   Rsquared        MAE 
# 15.3232101  0.9524683  7.6199822 

# error norm by row
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.000   3.929   7.162  10.976  11.893 349.378 

# summary(validation_small$error_distance)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.000   4.045   7.192  12.584  13.153 273.412 

# # filter >=-98
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.200   4.306   7.642  11.823  12.879 344.440 

# # filter >-90
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.000   4.050   7.354  11.796  12.828 276.832 

# 
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.000   3.720   6.896  10.336  11.303 357.028 

# function(x) var(x) <= 4 & max(x) <= -74, 
# Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# 0.2458   4.3431   7.7367  11.9778  12.7076 347.5786 

#filter >-85
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.000   4.050   7.694  12.483  13.344 279.543 

View(validation_small[which(validation_small$max > 40),])


hist(validation_small$error_distance)

errors_melt <- melt(validation_small$error_distance)

#
ggplot(errors_melt, aes(value)) + geom_histogram(binwidth = 5)

View(validation_small[which(validation_small$error_distance > 100),])

# plotlyTrainingData <- plot_ly(trainingData,
#                               x = trainingData$LATITUDE, 
#                               y = trainingData$LONGITUDE, z = trainingData$FLOOR, 
#                               color = trainingData$FLOOR, 
#                               colors = c('#BF382A','green' ,'#0C4B8E'),
#                               marker = list(size = 3)
#                               ) %>%
#   add_markers() %>%
#   layout(scene = list(xaxis = list(title = 'Latitude'),
#                       yaxis = list(title = 'Longitude'),
#                       zaxis = list(title = 'Floor')))

