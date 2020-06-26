#load packages
library(readr)
library(raster)
library(doParallel)
library(caTools)
library(foreach)
library(doParallel)
library(randomForest)
library(ranger)

##Corrected raster file of the full image stip from the satellite (~1600 km2, ~40GB)
library(raster)
#file<-"Q:/Satellite imagery test/workflow/sharpened/16SEP22191915_radiance_NNsharp_uinteger.dat"
#file<-"D:/Chiwawa test/ENVI Radiance Corrected/test classification/all_bands_subset.dat"

##read whole file into memory 
#################Big memory hog, but makes math much faster for index
chi<-readAll(brick(file))


#if it won't fit in memory just read in with brick
chi<-brick(file)
#check if loaded into memory
inMemory(chi)

# Rename layers in chi8 raster brick
names(chi) <- c('coastal', 'blue', 'green', 'yellow', 'red',
                'rededge', 'NIR1', 'NIR2' )

###normalized ratio equation for generating indices
nre_fun <- function(x, y) {
  nre <- (y - x) / (y + x)
  return(nre)
}


###generate some indices:
#ndvi=nir2-red/nir2+red
ndvi_fun <- function(x, y) {
  ndvi <- (y - x) / (y + x)
  return(ndvi)
}

#ndwi (also worldview water index)=coastal-nir2/coastal+nir2
ndwi_fun <- function(x, y) {
  ndwi <- (y - x) / (y + x)
  return(ndwi)
}

#ccci=(nir2-rede)/(nir2-rede)/((nir2-red)/nir2+red)
ccci_fun <- function(x, y, z) {
  ccci <- ((y - x) / (x + y))/((y - z) / (y + z))
  return(ccci)
}

##BAI=blue-NIR/(blue+NIR)
bai_fun <- function(x, y) {
  bai <- (y - x) / (y + x)
  return(bai)
}
##REI=NIR2-Blue/NIR2+Blue*NIR2
rei_fun <- function(x, y) {
  rei<- (y - x) / (y + x*y)
  return(rei)
}

#worldview built index (coastal-red edge)/(coastal+red edge)
wvbi_fun<- function(x, y) {
  wvbi <- (y - x) / (y + x)
  return(wvbi)
}

#NDSI normalized difference soil index (green-yellow)/(green+yellow)
ndsi_fun<- function(x, y) {
  ndsi <- (y - x) / (y + x)
  return(ndsi)
}




##Try combination of indices

chi_index<-stack(ndvi,ndwi,ccci,bai,rei,wvbi,ndsi)

names(chi_index) <- c('ndvi','ndwi','ccci','bai','rei','wvbi','ndsi')





# Read in training data ----
dfAll<-read.csv( file = "C:/Users/linds/NOAA/rf_training/data_raw/training_data_1M_sub.csv")
#dfAll<-read.csv( file = "Q:/Satellite imagery test/training_data.csv",header=TRUE)

indices <- matrix(data = NA, nrow = 55000, ncol = 36)

count <- 1
for (i in 1:8) {
  for (j in i:8) {
    indices[, count] = nre_fun(training_bc[i], training_bc[j])
    count = count + 1
  }
}
  


##Create indices
cl <- makeCluster(detectCores())
registerDoParallel(cl)
getDoParWorkers()
ndvi <- overlay(x=chi[[5]], y=chi[[8]], fun=ndvi_fun)
ndwi <- overlay(x=chi[[1]], y=chi[[8]], fun=ndwi_fun)
ccci <- overlay(x=chi[[6]], y=chi[[8]], z=chi[[5]], fun=ccci_fun)
bai <- overlay(x=chi[[2]], y=chi[[7]], fun=bai_fun)
rei <- overlay(x=chi[[2]], y=chi[[8]], fun=rei_fun)
wvbi <- overlay(x=chi[[6]], y=chi[[1]], fun=wvbi_fun)
ndsi <- overlay(x=chi[[4]], y=chi[[3]], fun=ndsi_fun)

stopCluster(cl)

#####Need to add "classname" back in as well, I haven't done that here yet.

chi_index<-stack(ndvi,ndwi,ccci,bai,rei,wvbi,ndsi)

names(chi_index) <- c('ndvi','ndwi','ccci','bai','rei','wvbi','ndsi' )



########################################################################
# Begin classification algorithms ----
######################################################################


## Fit Random Forest Model
### function to Select minimum number of samples from each category
dfAll <- na.omit(dfAll)
library(randomForest)
library(caret)
library(e1071)
set.seed(12)
undersample_ds <- function(x, classCol, nsamples_class) {
  for (i in 1:length(unique(x[, classCol]))) {
    class.i <- unique(x[, classCol])[i]
    if ((sum(x[, classCol] == class.i) - nsamples_class) != 0) {
      x <- x[-sample(which(x[, classCol] == class.i),
                     sum(x[, classCol] == class.i) - nsamples_class), ]
    }
  }
  return(x)
}
table(dfAll$Classname)
nsamples_class <- 5000

training_bc <- undersample_ds(dfAll, "Classname", nsamples_class)
training_bc$Classname <- as.factor(training_bc$Classname)
#write.csv(training_bc, file = "Q:/Satellite imagery test/training_data_1M_sub.csv",row.names=FALSE)
# set.seed(13)
# inTrain <- createDataPartition(y = training_bc$Classname,
#                                ## the outcome data are needed
#                                p = 0.90,
#                                ## The percentage of data in the
#                                ## training set
#                                list = FALSE)
# training <- training_bc[ inTrain, ]
# testing <- training_bc[-inTrain, ]
# 
# rf_fit <- ranger(Classname ~ coastal + blue + green + yellow + red + rededge + NIR1 + NIR2 +ccci + ndvi + ndwi + wv_water, data = training,
#                        mtry = 5, num.trees = 1500, verbose=TRUE, num.threads=12, importance="permutation", splitrule ="gini")
######Using randomforest package for easier prediction
library(foreach)
library(doParallel)
library(randomForest)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
getDoParWorkers()
set.seed(15)
inTrain <- createDataPartition(y = training_bc$Classname,
                               ## the outcome data are needed
                               p = .9,
                               ## The percentage of data in the
                               ## training set
                               list = FALSE)
training <- training_bc[ inTrain,]
testing <- training_bc[-inTrain,]


rf_fit<- randomForest(Classname ~ ndvi+ndwi+ccci+bai+rei+wvbi+ndsi, mtry = 4, ntree = 300, data = training)
stopCluster(cl)

# display results
print(rf_fit)
# plot(fit, scales = list(x = list(log = 10)))
# load("side_presence_contemp.RData")
rpartPred2 <- predict(rf_fit, testing)
confusionMatrix(rpartPred2, testing$Classname)
varImp(rf_fit)
importance(rf_fit)
LVQimportance <- varImp(rf_fit, scale = FALSE)
plot(LVQimportance, cex = 1.5)

# bankfull  widths from current parameters
save(rf_fit, file = "D:/Chiwawa test/rf_fit.RData")

load("D:/Chiwawa test/rf_fit.RData")

##Predict whole raster

###
library(randomForest)
preds_rf <- raster::predict(chi_index, model=rf_fit, na.rm=TRUE, type="response", progress='window')

#faster implementation
beginCluster(type="SOCK")
preds_rf <- clusterR(chi_index, predict, args = list(rf_fit),progress='window', type="response")
endCluster()

writeRaster(preds_rf, "D:/Chiwawa test/ENVI Radiance Corrected/test classification/chiwawa_RF_classification", format = "GTiff")



# Import Clip file to redice file size for prediction
clip_extent <- shapefile("chiwawa_clip_extent.shp")

chi_small <- crop(chi8, extent(clip_extent))

writeRaster(chi_small, "Chiwawa_index_small", format = "GTiff")



# Load file
chi_small <- brick('Chiwawa_index_small.tif')
names(chi_small) <- c("coastal", "blue", "green", "yellow", "red", "rededge", "NIR1", "NIR2", "ndvi57", "ndvi58", "ndvi47", "ndvi48", "ndvi37", "ndvi17", "ndvi68")

library(randomForest)

chi_small
beginCluster(8)
preds_rf <- clusterR(chi8, raster::predict, args = list(model = rf_fit))
endCluster()

# plot(preds_rf)

writeRaster(preds_rf, "Chiwawa prediction_rf_4", format = "GTiff")
# preds_rf

## Convert raster to Polygon