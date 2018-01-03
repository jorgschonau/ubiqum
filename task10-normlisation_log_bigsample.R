rm(list = ls())
#### Libraries ####
library(readr)
library(stats)
library(caret)
library(e1071)
library(dplyr)
library(C50)
library(lubridate)
library(anytime)
library(reshape2)

set.seed(456)

#### Data import, type conversions, basic pruning ####
trainingData <- read_csv(file="C:\\Users\\Jorg\\Desktop\\Ubiqum\\Task 10 - Wifi location\\UJIndoorLoc\\UJIndoorLoc\\trainingData.csv",
                         col_types = cols(BUILDINGID = col_character(), 
                                          FLOOR = col_character(), PHONEID = col_character(), 
                                          RELATIVEPOSITION = col_character(), 
                                          USERID = col_character()))

# --- data type conversion etc #
trainingData$FLOOR = as.factor(trainingData$FLOOR)
trainingData$LATITUDE = as.numeric(trainingData$LATITUDE)
trainingData$LONGITUDE = as.numeric(trainingData$LONGITUDE)
trainingData$BUILDINGID = as.factor(trainingData$BUILDINGID)
trainingData$SPACEID = as.factor(trainingData$SPACEID)
trainingData$RELATIVEPOSITION = as.factor(trainingData$RELATIVEPOSITION)
# removing useless variables (we need user id & device id for histograms, can delete later)
#trainingData$USERID <- NULL
#trainingData$PHONEID <- NULL
trainingData$TIMESTAMP <- NULL
# adding uniqe space id - might be useful for later 
trainingData$UNIQUESPACEID <- as.factor(paste(trainingData$BUILDINGID, trainingData$FLOOR, trainingData$SPACEID, trainingData$RELATIVEPOSITION, sep=""))
# adding ID no - might be useful for later (joining etc)
trainingData$ID <- seq.int(nrow(trainingData))
# --- removing waps with straight 100's 
trainingData <- trainingData[ - as.numeric(which(apply(trainingData, 2, var) == 0))]

training <- trainingData

#### PreProcessing, training sample etc ####

# Setting Threshold for Signal Strength 
#threshold = -95 #  The decibel threshold that you want to filter (i.e. remove all values less than -90 (-90 to -100) and replace with 100)

#View(trainingDataSmall) = trainingData[, !apply(trainingData== 100, 2, all)] #<----- Removes all solid 100 Columns

#View(training[,(ncol(training)-8):ncol(training)])
#View(training)

# replacing 100 by -100 (and every thing less than -100)
for(i in 1:465){
  training[which(training[,i] == 100 | training[,i] <= -100), i] = -100
}

#colnames(training)
# (and every thing less than -100)
# 
# for(i in 1:465){
#   training[which(training[,i] >= -100), i] = -100
# }


# removing columns with near zero variance (change < -96 to change) 
#training <- setdiff(training, training[,which(apply(training[,1:465],2,FUN = max) < -79 )])

training <- training[,-which(apply(training[,1:465],2,FUN = max) < -86 )]

#  Remove all columns with only 100's 
#training = training[, !apply(training == 100, 2, all)]

# Hist of wap signal strength distribution 
# training_melt <- melt(training[,1:(ncol(training)-8)])
# training_melt <- subset(training_melt, value != 100 )
# 
# ggplot(training_melt, aes(value)) + 
#   geom_histogram(binwidth = 1) 

# find rows with 0's - in both training and raw data set
#View(trainingData[which(trainingData[,1:465] == 0, arr.ind = TRUE)[,1],])
#View(trainingData[which(trainingData[,1:443] == 0, arr.ind = TRUE)[,1],])


# adding column with max value of each row
training$max <- apply(training[,1:(ncol(training)-10)], 1, FUN=max)

# view rows with max value of over -30 - over 500 rows
#View(training[which(training$max>=-25),])
lowmax <- training[which(training$max>=-24),]
#-24 best

# view rows with max value of 0 - about 118 rows
#View(training[which(training$max==0),])

# histogram
lowmax_melt <- melt(lowmax[,1:(ncol(lowmax)-8)])
lowmax_melt <- subset(lowmax_melt, value >=-30 )

ggplot(lowmax_melt, aes(value)) + 
  geom_histogram(binwidth = 1) 

# ----> exploring high strength signals
#plot(table(lowmax$BUILDINGID))
#plot(table(trainingData$PHONEID))

highmax <- training[which(training$max<=-75),]

phone22 <- training[which(training$PHONEID == 22),]

# ----> exploring high strength signals
plot(table(highmax$BUILDINGID))
plot(table(highmax$PHONEID))

plot(table(training$BUILDINGID))
plot(table(training$PHONEID))

plot(table(phone22$BUILDINGID))
plot(table(phone22$PHONEID))

# method 2 - anti-join (much simpler!)
training_less <- anti_join(training, lowmax, by = "ID")
#View(training_less)

# new! filtering out phone19
training_less <- training_less[-which(training_less$PHONEID == 19 | training_less$PHONEID == 7),]

#### Reverse values (into positives) ####
training_less_norm <- 100 + training_less[,1:(ncol(training)-11)]

# ----> Normalisation (for real) ----

max(training_less_norm)
normalize <- function(x) {
  return (x / 75) } # max(training_less_norm)

training_norm <- as.data.frame(t(apply(training_less_norm,1,normalize)))
# check
#View(training_norm)

# +1, log & division by log(2(max value)) to set scale 0 -1
training_norm <- training_norm + 1
#training_norm <- sqrt(training_norm)
training_norm <- log(training_norm)
training_norm <- training_norm/log(2)

#training_norm <- training_norm^2
###

# view rows with max value of over -30 - over 500 rows
#View(training_norm[which(training_norm$max>=0.9),])
#filter(training_norm, training_norm[,1:443] == 1)

# check histo
training_norm_melt <- melt(training_norm[,1:(ncol(training_norm)-8)])
training_norm_melt <- subset(training_norm_melt, value !=0)

ggplot(training_norm_melt, aes(value)) + 
  geom_histogram() 

# cbind
training_norm <- cbind(training_norm, training_less[,(ncol(training_less)-10):ncol(training_less)])
# replacing nan's (created by function above) by zeros
training_norm <- replace(training_norm, is.na(training_norm), 0)
# removing unnecessary columns
training_norm$max <- NULL
#### Modelling! ####

#training_less[,400:411]


#----> Create Training/Testing Partitions ----- 
sampleIndex = sample(1:nrow(training_norm), 5000)#<--------Size of your sample...smaller= faster but less accurate.

#View(training_norm)
# create Sample Dataframe called microTraining 
microTraining = training_norm[sampleIndex,]

inTraining <- createDataPartition(microTraining$BUILDINGID, p = .75, list = FALSE)
training <- na.omit(microTraining[inTraining,])
testing <- microTraining[-inTraining,]
#testing_copy <- shrunk[-inTraining,]
# adding row numbers to testing

#
#colnames(training[443:(ncol(training))])

#### Legend for Columns ####
#443 = Last WAP       ncol(trainingData)-8
#444 = Long           ncol(trainingData)-7
#445 = Lat            ncol(trainingData)-6
#446 = Floor          ncol(trainingData)-5
#447 = Building ID    ncol(trainingData)-4
#448 = Space ID       ncol(trainingData)-3
#449 = Rel Pos        ncol(trainingData)-2
#450 = UniquespaceID  ncol(trainingData)-1
#451 = ID             ncol(trainingData)-0

##### --------- STEP 1: Predicting Building ID using WAP ####
# fitcontrol 
fitControl<- trainControl(method="repeatedcv", number=10, repeats=3)

# -----> KNN Building -----

modelknnbld <- train(BUILDINGID ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-4))],
                     method = "knn", 
                     trControl = fitControl)
                     #preProcess = c("center","scale")) 
modelknnbld
# - only buildingid
# k  Accuracy   Kappa    
# 5  0.9971155  0.9954951

# ---- > SVM Building ----
modelsvmbld <- train(BUILDINGID ~ .,data = training[, c(1:(ncol(training)-8), (ncol(training)-4))],
                     method = "svmLinear2", 
                     trControl = fitControl)
                     #preProcess = c("center","scale")) 
modelsvmbld
# cost  Accuracy   Kappa   
#1.00  0.9980449  0.9969464

# ---- > GMB Building----
modelgmbbld = train(BUILDINGID ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-4))],
                    method = "gbm", 
                    trControl = fitControl)
                    #preProcess = c("center","scale") )
modelgmbbld 
# interaction.depth  n.trees  Accuracy   Kappa    
#3                  150      0.9986674  0.9979137

# ----- > C5.0 Building ----
modelC50bld <- train(BUILDINGID ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-4))], 
                     method = "C5.0",
                     trControl = fitControl)
                     #preProcess = c("center","scale"))
modelC50bld
# rules  FALSE   20      0.9962177  0.9940879

# ----- > RF Building ----
modelrfbld <- train(BUILDINGID ~ .,data = training[, c(1:(ncol(training)-8), (ncol(training)-4))],
                    method = "rf") 
                    #trControl = fitControl)
                    #preProcess = c("center","scale")) 
modelrfbld
# mtry  Accuracy   Kappa    
# 222   0.9910686  0.9860027
# 443   0.9862055  0.9783714

predrfbld <- predict(modelsvmbld, newdata = testing)

postResample(predrfbld, testing$BUILDINGID)

# RF
# Accuracy     Kappa 
# 0.9959920 0.9937566 
#
# c50
# Accuracy     Kappa 
# 0.9939880 0.9906467
# 
# GMB
# Accuracy    Kappa 
# 1        1 
# 0.9979960 0.9968822 
# 
# SVM
# Accuracy    Kappa 
# 1        1 
# 0.9959920 0.9937683
# 
# knn
# Accuracy     Kappa 
# 0.9859719 0.9782332

# -----> Winner for Building id: GMB or SVM----
# choose SVM as it takes less time

# postresampling: # 2   0.9982222  0.9972249

# Accuracy     Kappa 
# 0.9979960 0.9968701 

# overwriting testing$BUILDING with 
testing$BUILDINGID <- predrfbld

#### ---- STEP 2: Predicting Floor ####
# intermediate step: filter training & testing by building 
trainingbuild0 <- training[which(training$BUILDINGID == 0),]
trainingbuild1 <- training[which(training$BUILDINGID == 1),]
trainingbuild2 <- training[which(training$BUILDINGID == 2),]

testingbuild0 <- testing[which(testing$BUILDINGID == 0),]
testingbuild1 <- testing[which(testing$BUILDINGID == 1),]
testingbuild2 <- testing[which(testing$BUILDINGID == 2),]

# testing_copybuild0 <- testing_copy[which(testing_copy$BUILDINGID == 0),]
# testing_copybuild1 <- testing_copy[which(testing_copy$BUILDINGID == 1),]
# testing_copybuild2 <- testing_copy[which(testing_copy$BUILDINGID == 2),]

# converting floor to num and back to factor
trainingbuild0$FLOOR <- as.numeric(trainingbuild0$FLOOR)
trainingbuild1$FLOOR <- as.numeric(trainingbuild1$FLOOR)
trainingbuild2$FLOOR <- as.numeric(trainingbuild2$FLOOR)
trainingbuild0$FLOOR <- as.factor(trainingbuild0$FLOOR)
trainingbuild1$FLOOR <- as.factor(trainingbuild1$FLOOR)
trainingbuild2$FLOOR <- as.factor(trainingbuild2$FLOOR)

testingbuild0$FLOOR <- as.numeric(testingbuild0$FLOOR)
testingbuild1$FLOOR <- as.numeric(testingbuild1$FLOOR)
testingbuild2$FLOOR <- as.numeric(testingbuild2$FLOOR)
testingbuild0$FLOOR <- as.factor(testingbuild0$FLOOR)
testingbuild1$FLOOR <- as.factor(testingbuild1$FLOOR)
testingbuild2$FLOOR <- as.factor(testingbuild2$FLOOR)

#colnames(trainingbuild0[, c(1:(ncol(training)-12), (ncol(training)-9), (ncol(training)-8))] )

#View(trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))])

#398 = Last WAP       ncol(trainingData)-11
#401 = Floor          ncol(trainingData)-8
#402 = Building ID    ncol(trainingData)-7

# ----- > KNN floor -----
modelknnFloorBuild0 <- train(FLOOR ~ ., data = trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                             method = "knn", 
                             trControl = fitControl)
                             #preProcess = c("center","scale"))
modelknnFloorBuild0
# k  Accuracy   Kappa    
# 5  0.9538079  0.9382192
# 7  0.9461089  0.9279586
# 9  0.9341872  0.9120001

predknnfloorbuild0 <- predict(modelknnFloorBuild0,
                              newdata = testingbuild0)

postResample(predknnfloorbuild0, testingbuild0$FLOOR)

# Accuracy     Kappa 
# 0.7703704 0.6912351 

# ----> SVM floor Build 0-----

modelsvmDFloorBuild0 <- train(FLOOR ~ .,data = trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                              method = "svmLinear2", 
                              trControl = fitControl)
                              #preProcess = c("center","scale"))
modelsvmDFloorBuild0
# cost  Accuracy   Kappa    
# 0.25  0.9820019  0.9759226
# 0.50  0.9835708  0.9780242
# 1.00  0.9827956  0.9769893

predsvmfloorBuild0 <- predict(modelsvmDFloorBuild0, newdata = testingbuild0)
postResample(predsvmfloorBuild0, testingbuild0$FLOOR)
# Accuracy     Kappa 
# 1        1 

# ---- > GBM floor Build 0-----
modelGBMFloorBuild0 <- train(FLOOR ~ ., data = trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                             trControl = fitControl,
                             method = "gbm") 
                             #preProcess = c("center","scale"))
modelGBMFloorBuild0
# 3                  100      0.9544756  0.9391338

predfGMBloorbuild0 <- predict(modelGBMFloorBuild0, newdata = testingbuild0)
postResample (predfGMBloorbuild0, testingbuild0$FLOOR)
# Accuracy     Kappa 
# 0.9259259 0.9002733 

# ----> RF floor Build 0----
modelRFFloorBuild0 <- train(FLOOR ~ ., data = trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                            method = "rf",
                            trControl = fitControl)
                            #preProcess = c("center","scale"))
modelRFFloorBuild0 
# mtry  Accuracy   Kappa     
# 201   0.9383427  0.9169700
# 400   0.9210023  0.8935959
predfRFloorbuild0 <- predict(modelRFFloorBuild0, newdata = testingbuild0)
postResample(predfRFloorbuild0, testingbuild0$FLOOR)
# Accuracy     Kappa 
# 0.8962963 0.8599896

# ----> C50 floor Build 0 ----
modelC50floorBuild0 <- train(FLOOR ~ ., data = trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                             method = "C5.0",
                             trControl = fitControl)
                             #preProcess = c("center","scale"))
modelC50floor
# tree   FALSE   10      0.7752981  0.6988161
# tree   FALSE   20      0.7938176  0.7235486

predC50floorBuild0 <-predict(modelC50floorBuild0, newdata = testingbuild0)
postResample(predC50floorBuild0, testingbuild0$FLOOR)
# Accuracy     Kappa 
# 0.8666667 0.8206774 

# --------> Winner for build0: GBM
# postresample:  # 0.9259259 0.9002733  

testingbuild0$Floor <- predfGMBloorbuild0

# ---->> building 0 model: SVM ---- 

# ----> KNN floor Build 1 ----
modelknnFloorBuild1 <- train(FLOOR ~ ., data = trainingbuild1[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                             method = "knn", 
                             trControl = fitControl)
                             #preProcess = c("center","scale")) 
modelknnFloorBuild1
# k  Accuracy   Kappa    
# 5  0.9405300  0.9196175
# 7  0.9059892  0.8731660
# 9  0.8903764  0.8523361

predfknnfloorbuild1 <-predict(modelknnFloorBuild1, newdata = testingbuild1)
postResample(predfknnfloorbuild1, testingbuild1$FLOOR)
# Accuracy    Kappa 
# 0.9615385 0.9478916 

# ----> SVM floor Build 1 ----
modelsvmBLDFloorBuild1 <- train(FLOOR ~ .,data = trainingbuild1[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                                method = "svmLinear2", 
                                trControl = fitControl)
                                #preProcess = c("center","scale"))
modelsvmBLDFloorBuild1 
# cost  Accuracy   Kappa    
# 1.00  0.9707198  0.9603184

predsvmfloorbuild1 <-predict(modelsvmBLDFloorBuild1 , newdata = testingbuild1)
postResample(predsvmfloorbuild1, testingbuild1$FLOOR)
# Accuracy     Kappa 
# 0.9692308 0.9582531

# ----> GBM floor Build 1 ----
modelGBMFloorBuild1 <- train(FLOOR ~ ., data = trainingbuild1[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                             method = "gbm",
                             trControl = fitControl)
                             #preProcess = c("center","scale"))
modelGBMFloorBuild1
#3                  100      0.9370825  0.9147728

predGBMFloorBuild1 <- predict(modelGBMFloorBuild1, newdata = testingbuild1)
postResample(predGBMFloorBuild1, testingbuild1$FLOOR)
# Accuracy     Kappa 
# 0.9615385 0.9479167

# ----> RF floor Build1 ----
modelRFFloorBuild1 <- train(FLOOR ~ ., data = trainingbuild1[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                            method = "rf", 
                            trControl = fitControl)
                            #preProcess = c("center","scale"))
modelRFFloorBuild1
# mtry  Accuracy   Kappa    
# 223   0.9415315  0.92087091
# 445   0.9311828  0.90705874

predmodelRFFloorBuild1 <- predict(modelRFFloorBuild1, newdata = testingbuild1)
postResample(predmodelRFFloorBuild1, testingbuild1$FLOOR)
# Accuracy     Kappa 
# 0.9538462 0.9374549  

# ----> C50 floor Build1 ----
modelC50floorBuild1 <- train(FLOOR ~ ., data = trainingbuild1[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                             method = "C5.0",
                             trControl = fitControl)
                             #preProcess = c("center","scale"))
modelC50floorBuild1
# tree   FALSE   20      0.9319609  0.9079839

predC50floorBuild1 <- predict(modelC50floorBuild1, newdata = testingbuild1)
postResample(predC50floorBuild1, testingbuild1$FLOOR)
# Accuracy     Kappa 
# 0.9461538 0.9271184

# -----> Winner for build1: svm, predsvmfloorBuild1
# ---- postresample: # 1.00  0.9707198  0.9603184

testingbuild1$FLOOR <- predC50floorBuild1

# -----> Building 2
# ----> KNN floor Build2 ----
modelknnFloorBuild2 <- train(FLOOR ~ ., data = trainingbuild2[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                             method = "knn", 
                             trControl = fitControl)
                             #preProcess = c("center","scale")) 
modelknnFloorBuild2
# k  Accuracy   Kappa    
# 5  0.9656404  0.9556802
# 7  0.9603189  0.9487392
# 9  0.9603262  0.9487318

predknnFloorBuild2 <- predict(modelknnFloorBuild2, newdata = testingbuild2)
postResample(predknnFloorBuild2, testingbuild2$FLOOR)
# Accuracy     Kappa 
# 0.9825328 0.9774462 

# ----> SVM floor Build2 ----
modelsvmBLDFloorbuild2 <- train(FLOOR ~ .,data = trainingbuild2[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                                method = "svmLinear2", 
                                trControl = fitControl)
                                #preProcess = c("center","scale"))
modelsvmBLDFloorbuild2
# cost  Accuracy   Kappa    
# 0.25  0.9850199  0.9807134
# 0.50  0.9849988  0.9806789
# 1.00  0.9864625  0.9825704

predsvmFloorbuild2 <- predict(modelsvmBLDFloorbuild2, newdata = testingbuild2)
postResample(predsvmFloorbuild2, testingbuild2$FLOOR)
# Accuracy     Kappa 
# 1        1 

# ----> GBM floor Build2 ----
modelGBMFloorbuild2 <- train(FLOOR ~ ., data = trainingbuild2[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                             method = "gbm",
                             trControl = fitControl)
                             #preProcess = c("center","scale"))
modelGBMFloorbuild2
# 3                  100      0.9452185  0.9300994
# 3                  150      0.9514165  0.9379996

predGBMFloorbuild2 <- predict(modelGBMFloorbuild2, newdata = testingbuild2)
postResample(predGBMFloorbuild2, testingbuild2$FLOOR)
# Accuracy     Kappa 
# 0.9741379 0.9668910 

# ----> RF floor Build2 ----
modelRFFloorbuild2 <- train(FLOOR ~ ., data = trainingbuild2[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                            method = "rf",
                            trControl = fitControl)
                            #preProcess = c("center","scale"))
modelRFFloorbuild2
# 201   0.9422547  0.92611660
# 400   0.9201210  0.89772634

predRFFloorbuild2 <- predict(modelRFFloorbuild2, newdata = testingbuild2)
postResample(predRFFloorbuild2, testingbuild2$FLOOR)
# Accuracy     Kappa 
# 0.9698276 0.9613232 

# ----> C50 floor Build2 ----
modelC50floorbuild2 <- train(FLOOR ~ ., data = trainingbuild2[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                             method = "C5.0",
                             trControl = fitControl)
                             #preProcess = c("center","scale"))
modelC50floorbuild2
# tree   FALSE   10      0.8922023  0.8623645
# tree   FALSE   20      0.9042603  0.8776709

predC50floorbuild2  <- predict(modelC50floorbuild2, newdata = testingbuild2)
postResample(predC50floorbuild2, testingbuild2$FLOOR)
# Accuracy     Kappa 
# 0.9353448 0.9175746 

# -----> winner for build 2: GBM, modelGBMFloorbuild2 ----
# -----> postresample: # 0.9741379 0.9668910 

testingbuild2$FLOOR <- predGBMFloorbuild2

# to recapitulate: build0: GBM, build1: C5.0, build 2: GBM

# ----> inserting predictions into testing ----
# converting to num and converting back to factor (not sure if actually necessary)
testingbuild0$FLOOR <- as.numeric(testingbuild0$FLOOR)
testingbuild1$FLOOR <- as.numeric(testingbuild1$FLOOR)
testingbuild2$FLOOR <- as.numeric(testingbuild2$FLOOR)

predfGMBloorbuild0_num <- as.numeric(predfGMBloorbuild0)
predC50floorBuild1_num <- as.numeric(predC50floorBuild1)
predGBMFloorbuild2_num <- as.numeric(predGBMFloorbuild2)

testingbuild0$FLOOR <- predfGMBloorbuild0_num
testingbuild1$FLOOR <- predC50floorBuild1_num
testingbuild2$FLOOR <- predGBMFloorbuild2_num

testingbuildall <- rbind(testingbuild0, testingbuild1, testingbuild2)

testingbuildall <- rename(testingbuildall, predFLOOR = FLOOR)

testingbuildall$predFLOOR <- as.factor(testingbuildall$predFLOOR)

levels(testingbuildall$predFLOOR) <- c(0:4)

testing_merge <- merge(testing, testingbuildall[,c("predFLOOR","ID")], by="ID")

# inserting predictions for Floor

testing_merge$FLOOR <- NULL
testing_merge$FLOOR <- testing_merge$predFLOOR
testing_merge$predFLOOR <- NULL
testing <- testing_merge

# ----> Final result for prediction of floors by building: ----
postResample(testing_merge$predFLOOR, testing_merge$FLOOR)
# Accuracy     Kappa 
# 0.9458918 0.9298040 

#### STEP 2B: Predicting Floors in one go ####

# ----> KNN floor Build all ----
modelknnFloorBuildAll <- train(FLOOR ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                               method = "knn", 
                               trControl = fitControl)
                               #preProcess = c("center","scale")) 
modelknnFloorBuildAll

predknnBLDFloorbuilAll <- predict(modelknnFloorBuildAll, newdata = testing)
postResample(predknnBLDFloorbuilAll, testing$FLOOR)
# Accuracy     Kappa 
# 5  0.9560101  0.9425757
# 7  0.9460350  0.9295371

# ----> SVM floor Build all ----
modelsvmBLDFloorbuilAll <- train(FLOOR ~ .,data = training[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                                 method = "svmLinear2", 
                                 trControl = fitControl)
                                 #preProcess = c("center","scale"))
modelsvmBLDFloorbuilAll 
# cost  Accuracy   Kappa    
# 0.50  0.9807012  0.9748096
# 1.00  0.9807011  0.9748050

predsvmBLDFloorbuilAll <- predict(modelsvmBLDFloorbuilAll, newdata = testing)
postResample(predsvmBLDFloorbuilAll, testing$FLOOR)

# Accuracy     Kappa 
# 0.9839679 0.9790721 

# ----> GBM floor Build all ----
modelGBMFloorAll <- train(FLOOR ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                          method = "gbm",
                          trControl = fitControl)
                          #preProcess = c("center","scale"))
modelGBMFloorAll
#3                  150      0.9509146  0.9358997

predGBMFloorBuildAll <- predict(modelGBMFloorAll, newdata = testing)
postResample(predGBMFloorBuildAll, testing$FLOOR)
# Accuracy     Kappa 
# 0.9038076 0.8751752 

# ----> RF Floor BuildAll ----
modelRFFloorBuildAll <- train(FLOOR ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                              method = "rf", 
                              trControl = fitControl)
                              #preProcess = c("center","scale"))
modelRFFloorBuildAll
# mtry  Accuracy   Kappa    
# 201   0.9313503  0.9109849
# 400   0.9166951  0.8919663

predmodelRFFloorBuildAll <- predict(modelRFFloorBuildAll, newdata = testing)
postResample(predmodelRFFloorBuildAll, testing$FLOOR)
# Accuracy     Kappa 
# 0.9238477 0.9012802

# ----> C50 floor BuildAll ----
modelC50floorBuildAll <- train(FLOOR ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-5), (ncol(training)-4))],
                               method = "C5.0",
                               trControl = fitControl)
                               #preProcess = c("center","scale"))
modelC50floorBuildAll
# tree   FALSE   10      0.8940746  0.8626498
# tree   FALSE   20      0.9049726  0.8767837

predC50floorBuildAll <- predict(modelC50floorBuildAll, newdata = testing)
postResample(predC50floorBuildAll, testing$FLOOR)
# Accuracy     Kappa 
# 0.9098196 0.8833452 

# ----> Final Results: SVM best ----
postResample(predsvmBLDFloorbuilAll, testing$FLOOR)
# Accuracy     Kappa 
# 0.9839679 0.9790721 

# inserting predictions

testing$FLOOR <- predsvmBLDFloorbuilAll

#### STEP 3 Model Comparison for Lat ####

#443 = Last WAP       ncol(trainingData)-8
#444 = Long           ncol(trainingData)-7
#445 = Lat            ncol(trainingData)-6
#446 = Floor          ncol(trainingData)-5
#447 = Building ID    ncol(trainingData)-4
#448 = Space ID       ncol(trainingData)-3
#449 = Rel Pos        ncol(trainingData)-2
#450 = UniquespaceID  ncol(trainingData)-1
#451 = ID             ncol(trainingData)-0

#View(training[, c(1:(ncol(training)-8), (ncol(training)-9), (ncol(training)-8), (ncol(training)-4))])

#-----> KNN for LAT ----
modelknnLAT <- train(LATITUDE ~ ., data = training[, c(1:(ncol(training)-10), (ncol(training)-8), (ncol(training)-7), (ncol(training)-6))],
                     method = "knn",
                     trControl = fitControl)
                     #preProcess = c("center","scale")) 

modelknnLAT_nofloor <- train(LATITUDE ~ ., data = training[, c(1:(ncol(training)-10), (ncol(training)-8), (ncol(training)-6))],
                     method = "knn",
                     trControl = fitControl)

colnames(training[, c(1:(ncol(training)-10), (ncol(training)-8), (ncol(training)-6))])

#preProcess = c("center","scale")) 

modelknnLAT # 5000 sample
# k  RMSE      Rsquared   MAE     
# 5  5.301781  0.9936573  3.282928
# 7  5.752935  0.9925486  3.645021
# 9  6.084883  0.9916749  3.927813

modelknnLAT_nofloor # 5000 sample
# k  RMSE      Rsquared   MAE     
# 5  5.438689  0.9933433  3.281984
# 7  5.633533  0.9928812  3.563976
# 9  5.852563  0.9923258  3.802395
# with squareroot
# k  RMSE      Rsquared   MAE     
# 5  6.161359  0.9916045  3.816241
# 7  6.415203  0.9909231  4.052574
# 9  6.607062  0.9903798  4.235481
# with + 1 squareroot
# k  RMSE      Rsquared   MAE     
# 5  5.479421  0.9932528  3.351813
# 7  5.743821  0.9926017  3.677646
# 9  5.973002  0.9920102  3.901693
# with +1& log
# k  RMSE      Rsquared   MAE     
# 5  5.383689  0.9933939  3.360769
# 7  5.609132  0.9928352  3.625895
# 9  5.928720  0.9920146  3.883208
# lowmax 35
# k  RMSE      Rsquared   MAE     
# 5  5.960951  0.9918931  3.504846
# 7  6.216828  0.9912495  3.807887
# 9  6.482027  0.9905283  4.055802
# without phoneid 19
# k  RMSE      Rsquared   MAE     
# 5  5.583676  0.9929080  3.357030
# 7  5.780181  0.9924317  3.613593
# 9  5.982738  0.9919067  3.837908
# cutoff point > 90
# k  RMSE      Rsquared   MAE     
# 5  5.472082  0.9930540  3.358525
# 7  5.764891  0.9922932  3.650277
# 9  5.953718  0.9918225  3.879871
# cutoff point > 79
k  RMSE      Rsquared   MAE     
5  5.790318  0.9926497  3.508226
7  6.172745  0.9916685  3.869449
9  6.424175  0.9909955  4.138909


#7.48 ...better without log

predknnLAT_nofloor <- predict(modelknnLAT_nofloor, newdata = testing)
postResample(predknnLAT_nofloor, testing$LATITUDE)
# RMSE Rsquared      MAE 
# 5.907392 0.992402 3.502717

#-----> SVM for LAT ----
modelsvmLAT <- train(LATITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-5), (ncol(training)-4))],
                     method = "svmLinear2",
                     trControl = fitControl)
                     #preProcess = c("center","scale")) 
modelsvmLAT
# cost  RMSE      Rsquared   MAE     
# 0.25  25.40792  0.9015122  17.87983
# 0.50  19.80602  0.9321160  13.72427
# 1.00  15.81796  0.9524770  11.15526

predsvmLAT <- predict(modelsvmLAT, newdata = testing)
postResample(predsvmLAT, testing$LATITUDE)
# RMSE   Rsquared        MAE 
# 17.7252663  0.9296107 12.7239078 

#-----> GBM for LAT ----
modelGBMlat = train(LATITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-5), (ncol(training)-4))],
                    method = "gbm",
                    trControl = fitControl)
                    #preProcess = c("center","scale") )


modelGBMlat_noFloor = train(LATITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-9), (ncol(training)-4))],
                            method = "gbm", 
                            trControl = fitControl,
                            preProcess = c("center") )

modelGBMlat 
#0.4  3          0.8               1.00       150      12.46136  0.9653520   8.991370
predgbmLAT <- predict(modelGBMlat, newdata = testing)
postResample(predgbmLAT, testing$LATITUDE)
# RMSE   Rsquared        MAE 
# 11.5193070  0.9705073  8.0904330 

modelGBMlat_noFloor
# 3                  150      12.87636  0.9632442   9.868676
predgbmLAT_noFloor <- predict(modelGBMlat_noFloor, newdata = testing)
postResample(predgbmLAT_noFloor, testing$LATITUDE)
#RMSE   Rsquared        MAE 
#12.9156404  0.9627073  9.5783507 

#------> RF for LAT ----
modelRFlat <- train(LATITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-5), (ncol(training)-4))],
                    method = "rf",
                    trControl = fitControl),
                    ntree = 1)
modelRFlat 
# 225   15.07612  0.9508877   8.925524

predrfLAT <- predict(modelRFlat, newdata = testing)
postResample(predrfLAT, testing$LATITUDE)
# RMSE   Rsquared        MAE 
# 18.2082158  0.9272415 10.4987877 

#------> XGB Linear for LAT ----
modelxgblinlat = train(LATITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-5), (ncol(training)-4))],
                       method = "xgbLinear", 
                       trControl = fitControl)
                       #preProcess = c("center","scale") )
modelxgblinlat
# 0e+00   1e-04  100       9.816994  0.9791242  6.847718
# 0e+00   1e-04  150       9.805861  0.9791832  6.828599

xgblinlatLAT <- predict(modelxgblinlat, newdata = testing)
postResample(xgblinlatLAT, testing$LATITUDE)
# RMSE   Rsquared        MAE 
# 9.5956543 0.9806587 6.4490496 
#------> XGB Dart for LAT ----
modelxgbdartlat = train(LATITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-5), (ncol(training)-4))],
                        method = "xgbDART",
                        trControl = fitControl)
                        #preProcess = c("center","scale") )
predxgbdartLAT <- predict(modelxgbdartlat, newdata = testing)
postResample(predxgbdartLAT, testing$LATITUDE)
# RMSE    Rsquared         MAE 
# 4228.950739    0.971731 4228.935772 
#------> XGB Tree for LAT ----

modelxgbtreelat = train(LATITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-5), (ncol(training)-4))],
                        method = "xgbTree",
                        trControl = fitControl)
                        #preProcess = c("center","scale") )
predxgbtreelat <- predict(modelxgbtreelat, newdata = testing)
postResample(predxgbtreelat, testing$LATITUDE)
# RMSE   Rsquared        MAE 
# 11.2502662  0.9735204  7.6634710 
#------> Rpart for LAT ----
modelrpartlat<- train(LATITUDE ~., data = training[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-5), (ncol(training)-4))],
                      method = "rpart",
                      trControl = fitControl),
                      tuneLength = 10)

modelrpartlat
# 0.006963939  20.54208  0.9058014  15.10281
# 0.007457307  20.91549  0.9023966  15.37836
predrpartlat <- predict(modelrpartlat, newdata = testing)
postResample(predrpartlat, testing$LATITUDE)
# RMSE   Rsquared        MAE 
# 21.9265823  0.8923889 16.2755154 

#------> Final results for LAT ----
# knn the best:
# RMSE   Rsquared        MAE 
# 10.3653208  0.9766321  6.6880848 

# removing Floor has a minor negative impact on LAT/LONG preditions


#### ---- STEP 3b: Predicting lat by building ####
# intermediate step: filter training & testing by building 
trainingbuild0 <- training[which(training$BUILDINGID == 0),]
trainingbuild1 <- training[which(training$BUILDINGID == 1),]
trainingbuild2 <- training[which(training$BUILDINGID == 2),]

testingbuild0 <- testing[which(testing$BUILDINGID == 0),]
testingbuild1 <- testing[which(testing$BUILDINGID == 1),]
testingbuild2 <- testing[which(testing$BUILDINGID == 2),]

# testing_copybuild0 <- testing_copy[which(testing_copy$BUILDINGID == 0),]
# testing_copybuild1 <- testing_copy[which(testing_copy$BUILDINGID == 1),]
# testing_copybuild2 <- testing_copy[which(testing_copy$BUILDINGID == 2),]

# converting floor to num and back to factor
trainingbuild0$FLOOR <- as.numeric(trainingbuild0$FLOOR)
trainingbuild1$FLOOR <- as.numeric(trainingbuild1$FLOOR)
trainingbuild2$FLOOR <- as.numeric(trainingbuild2$FLOOR)
trainingbuild0$FLOOR <- as.factor(trainingbuild0$FLOOR)
trainingbuild1$FLOOR <- as.factor(trainingbuild1$FLOOR)
trainingbuild2$FLOOR <- as.factor(trainingbuild2$FLOOR)

testingbuild0$FLOOR <- as.numeric(testingbuild0$FLOOR)
testingbuild1$FLOOR <- as.numeric(testingbuild1$FLOOR)
testingbuild2$FLOOR <- as.numeric(testingbuild2$FLOOR)
testingbuild0$FLOOR <- as.factor(testingbuild0$FLOOR)
testingbuild1$FLOOR <- as.factor(testingbuild1$FLOOR)
testingbuild2$FLOOR <- as.factor(testingbuild2$FLOOR)

colnames(trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))] )
#View(trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))])

#443 = Last WAP       ncol(trainingData)-8
#443 = LAT            ncol(trainingData)-6
#402 = Building ID    ncol(trainingData)-4

# ----- > Build 0 -----
modelknnlatBuild0 <- train(LATITUDE ~ ., data = trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                             method = "knn", 
                             trControl = fitControl)
modelknnlatBuild0
# k  RMSE      Rsquared   MAE     
# 5  3.999528  0.9846401  2.736261

modelxgbLinearlatBuild0 <- train(LATITUDE ~ ., data = trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                             method = "xgbLinear", 
                             trControl = fitControl)
modelxgbLinearlatBuild0
#1e-04   0e+00  150      5.320710  0.9723291  3.678959

modelgbmlatBuild0 <- train(LATITUDE ~ ., data = trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                             method = "gbm", 
                             trControl = fitControl)
modelgbmlatBuild0
#3                  150       6.574656  0.9577562  4.926615

modelrflatBuild0 <- train(LATITUDE ~ ., data = trainingbuild0[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                             method = "rf", 
                             trControl = fitControl)
modelrflatBuild0

# ----- > Build 1 -----
modelknnlatBuild1 <- train(LATITUDE ~ ., data = trainingbuild1[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                           method = "knn", 
                           trControl = fitControl)
modelknnlatBuild1
# k  RMSE      Rsquared   MAE     
# 5  7.389421  0.9571203  4.648130

modelxgbLinearlatBuild1 <- train(LATITUDE ~ ., data = trainingbuild1[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                                 method = "xgbLinear", 
                                 trControl = fitControl)
modelxgbLinearlatBuild1
#0e+00   0e+00  100      7.416450  0.9569547  5.042954

modelgbmlatBuild1 <- train(LATITUDE ~ ., data = trainingbuild1[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                           method = "gbm", 
                           trControl = fitControl)
modelgbmlatBuild1
#3                  150       8.901315  0.9382330   6.590604

modelrflatBuild1 <- train(LATITUDE ~ ., data = trainingbuild1[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                          method = "rf", 
                          trControl = fitControl)
modelrflatBuild1

# ----> winner for lat build 1: knn ----

# ----- > Build 2 -----
modelknnlatBuild2 <- train(LATITUDE ~ ., data = trainingbuild2[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                           method = "knn", 
                           trControl = fitControl)
modelknnlatBuild2
# k  RMSE      Rsquared   MAE     
#5  5.366058  0.9620530  3.313243

modelxgbLinearlatBuild2 <- train(LATITUDE ~ ., data = trainingbuild2[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                                 method = "xgbLinear", 
                                 trControl = fitControl)
modelxgbLinearlatBuild2
#0e+00   1e-04  150      6.996357  0.9360575  4.806681

modelgbmlatBuild2 <- train(LATITUDE ~ ., data = trainingbuild2[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                           method = "gbm", 
                           trControl = fitControl)
modelgbmlatBuild2
#3                  150       8.484070  0.9059526   6.315885

modelrflatBuild2 <- train(LATITUDE ~ ., data = trainingbuild2[, c(1:(ncol(training)-8), (ncol(training)-6), (ncol(training)-4))],
                          method = "rf", 
                          trControl = fitControl)
modelrflatBuild2

#### STEP 4 Model Comparison for Long ####

#### Legend for Columns ####
#443 = Last WAP       ncol(trainingData)-8
#444 = Long           ncol(trainingData)-7
#445 = Lat            ncol(trainingData)-6
#446 = Floor          ncol(trainingData)-5
#447 = Building ID    ncol(trainingData)-4
#448 = Space ID       ncol(trainingData)-3
#449 = Rel Pos        ncol(trainingData)-2
#450 = UniquespaceID  ncol(trainingData)-1
#451 = ID             ncol(trainingData)-0


#View(trainingData[,c((ncol(trainingData)-11):(ncol(trainingData)-10))])

#-----> KNN for LONG ----
modelknnLONG <- train(LONGITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-7), (ncol(training)-5), (ncol(training)-4))],
                      method = "knn",
                      trControl = fitControl)
                      #preProcess = c("center","scale")) 
#modelknnLONG
# k  RMSE      Rsquared   MAE     
# 5  11.68885  0.9694487  7.314443
# 7  12.00144  0.9679742  7.616802
# 9  12.30588  0.9665517  7.924764

predknnLONG <- predict(modelknnLONG, newdata = testing)
#postResample(predknnLONG, testing$LONGITUDE)
# RMSE   Rsquared        MAE 
# 10.3653208  0.9766321  6.6880848 

#-----> SVM for LONG ----
modelsvmLONG <- train(LONGITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-7), (ncol(training)-5), (ncol(training)-4))],
                      method = "svmLinear2",
                      trControl = fitControl)
                      #preProcess = c("center","scale")) 
#modelsvmLONG
# cost  RMSE      Rsquared   MAE     
# 0.25  31.79296  0.9345022  22.13165
# 0.50  31.43406  0.9364364  21.77964
# 1.00  29.67916  0.9434573  20.83545

predsvmLONG <- predict(modelsvmLONG, newdata = testing)
postResample(predsvmLONG, testing$LONGITUDE)
# RMSE   Rsquared        MAE 
# 23.2410006  0.9642946 16.2675286 

#-----> GBM for LONG ----
modelGBMLONG = train(LONGITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-7), (ncol(training)-5), (ncol(training)-4))],
                     method = "gbm",
                     trControl = fitControl)
                     #preProcess = c("center","scale") )

modelGBMLONG 
# 3                  150      15.95772  0.9834484  12.19954predgbmLONG <- predict(modelGBMLONG, newdata = testing)

predGBMLONG <- predict(modelsvmLONG, newdata = testing)
postResample(predGBMLONG, testing$LONGITUDE)

# RMSE   Rsquared        MAE 
# 23.2410006  0.9642946 16.2675286 

#------> RF for LONG ----
modelRFLONG <- train(LONGITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-7), (ncol(training)-5), (ncol(training)-4))],
                     method = "rf",
                     trControl = fitControl),
                     ntree = 1)
#modelRFLONG 
#203   16.17105  0.9413705   9.850733

predrfLONG <- predict(modelRFLONG, newdata = testing)
postResample(predrfLONG, testing$LONGITUDE)
# RMSE   Rsquared        MAE 
# 18.4663117  0.9775114 10.4078052

#------> XGB Linear for LONG ----
modelxgblinLONG = train(LONGITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-7), (ncol(training)-5), (ncol(training)-4))],
                        method = "xgbLinear",
                        trControl = fitControl)
                        #preProcess = c("center","scale") )
#modelxgblinLONG
#0e+00   0e+00  150      10.90892  0.9731534  7.759733

predxgblinLONG <- predict(modelxgblinLONG, newdata = testing)
postResample(predxgblinLONG, testing$LONGITUDE)
# RMSE   Rsquared        MAE 
# 13.0173691  0.9887432  7.5646332 
#------> XGB Dart for LONG ----
modelxgbdartLONG = train(LONGITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-7), (ncol(training)-5), (ncol(training)-4))],
                         method = "xgbDART", 
                         trControl = fitControl)
                         #preProcess = c("center","scale"))

predxgbdartLONG <- predict(modelxgbdartLONG, newdata = testing)
postResample(predxgbdartLONG, testing$LONGITUDE)
# RMSE   Rsquared        MAE 
# 14.8234289  0.9854645  9.6801863 

#------> XGB Tree for LONG ----
modelxgbtreeLONG = train(LONGITUDE ~ ., data = training[, c(1:(ncol(training)-8), (ncol(training)-7), (ncol(training)-5), (ncol(training)-4))],
                         method = "xgbTree",
                         trControl = fitControl)
                         #preProcess = c("center","scale"))

predxgbtreeLONG <- predict(modelxgbtreeLONG, newdata = testing)
#postResample(predxgbtreeLONG, testing$LONGITUDE)
# RMSE   Rsquared        MAE 
# 11.4730101  0.9704966  8.1653071 

#------> Final Results for LONG ----
#KNN works best
#predknnLONG <- predict(modelknnLONG, newdata = testing)
#postResample(predknnLONG, testing$LONGITUDE)
# RMSE   Rsquared        MAE 
# 10.3653208  0.9766321  6.6880848 


#### Step 5 Running on models on the entire data #####

#----> creating big testing df
# --- recyling stuff from proprecess

#----------- Remove all columns with only 100's 
testingall = trainingData[, !apply(trainingData == 100, 2, all)] #<----- Removes all solid 100 Columns

#----------- > Replace all values less than the rheshold DB with 100 --------------------

for(i in 1:(ncol(testingall)-11)) #<----------  Removes all values for decibel threshold
{
  testingall[which(testingall[,i] < threshold ), i] = 100
}

#----------- Remove all columns with only 100's 
testingall =  testingall[, !apply( testingall == 100, 2, all)]

testallonlywaps <- testingall[, 1:(ncol(testingall)-11)]

#----> Predicting Building ----

predmodelknnbld_all <- predict(modelknnbld,
                               newdata = testallonlywaps)
testallonlywaps$BUILDINGID <- predmodelknnbld_all

#----> Predicting Floor, without splitting by building (too time consuming) -----

predmodelRFFloorBuildAll_bigtest <- predict(modelRFFloorBuildAll, newdata = testallonlywaps)
testallonlywaps$FLOOR <- predmodelRFFloorBuildAll_bigtest

#----> Predicting LAT -----

predknnLAT_all <- predict(modelknnLAT, newdata = testallonlywaps)
testallonlywaps$LATITUDE <- predknnLAT_all


View(head(testallonlywaps))

#----> Predicting LONG -----

predknnLONG_all <- predict(modelknnLONG, newdata = testallonlywaps)
testallonlywaps$LONGITUDE <- predknnLONG_all

#----> Scatter Plot for comparing pred vs actual coordinates ----

coordsreal <- testingall[,c(445,444)]
coordspreds <- testallonlywaps[,446:447]

coordspreds <- rename(coordspreds, predLATITUDE = LATITUDE)
coordspreds <- rename(coordspreds, predLONGITUDE = LONGITUDE)


# -combined scatter

ggplot() + 
  geom_point(data = coordsreal, aes(x = LONGITUDE, y = LATITUDE), color = "black", size = 1) +
  geom_point(data = coordspreds, aes(x = predLONGITUDE, y =predLATITUDE) , color = "blue", size = 0.4, alpha = 0.05, shape = 4) +
  xlab('Latitude') +
  ylab('Longitude')+
  labs(title = "Actual (black) vs Predicted Data (blue)")




library(plotly)
plotlyTrainingData <- plot_ly(trainingData, x = trainingData$LATITUDE, y = trainingData$LONGITUDE, z = trainingData$FLOOR, color = trainingData$FLOOR, colors = c('#BF382A','green' ,'#0C4B8E')) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Latitude'),
                      yaxis = list(title = 'Longitude'),
                      zaxis = list(title = 'Floor')))

plotlyTrainingData


View(training_norm)
x2 <- na.omit(x2)

# cbind ...
training_norm <- cbind(training_less_norm, training_less[,444:452])

View(trainingData[5,])
View(training_norm)


# miguel's code






