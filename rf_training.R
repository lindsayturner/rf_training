#load packages
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
dfAll<-read.csv( file = "C:/Users/linds/NOAA/rf_training/data_raw/training_data_1M_sub.csv",header=T)
#dfAll<-read.csv( file = "Q:/Satellite imagery test/training_data.csv",header=TRUE)

### function to Select minimum number of samples from each category
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
nsamples_class <- 10000

training_bc <- undersample_ds(dfAll, "Classname", nsamples_class)
training_bc$Classname <- as.factor(training_bc$Classname)


# Use top 5/10/15 indices from tests

indices <- matrix(data = NA, nrow = 110000, ncol = 5)

green.red <- nre_fun(training_bc[3], training_bc[5])
blue.coastal <- nre_fun(training_bc[2], training_bc[1])
NIR2.yellow <- nre_fun(training_bc[8],training_bc[4])
NIR1.red <- nre_fun(training_bc[7],training_bc[5])
rededge.yellow <- nre_fun(training_bc[6],training_bc[4])

red.NIR2 <- nre_fun(training_bc[5],training_bc[8])
rededge.NIR2 <- nre_fun(training_bc[6],training_bc[8])
rededge.NIR1 <- nre_fun(training_bc[6],training_bc[7])
green.NIR1 <- nre_fun(training_bc[3],training_bc[7])
green.NIR2 <- nre_fun(training_bc[3],training_bc[8])

rededge.green <- nre_fun(training_bc[6],training_bc[3])
rededge.red <- nre_fun(training_bc[6],training_bc[5])
yellow.NIR1 <- nre_fun(training_bc[4],training_bc[7])
NIR2.blue <- nre_fun(training_bc[8],training_bc[2])
blue.red <- nre_fun(training_bc[2],training_bc[5])

indices <- cbind(green.red, blue.coastal, NIR2.yellow, NIR1.red, rededge.yellow)
names(indices) <- c('green.red', 'blue.coastal', 'NIR2.yellow', 'NIR1.red', 'rededge.yellow')
indices_df<-as.data.frame(indices)
indices_df <- indices_df * 10000
indices_df$Classname<-training_bc$Classname
head(indices_df)

# Generate all indices using training data

indices <- matrix(data = NA, nrow = 110000, ncol = 64)

count <- 1
col_names <- character(64)
for (i in 1:8) {
  for (j in 1:8) {
    indices[, count] <- nre_fun(training_bc[,i], training_bc[,j]) * 10000
    col_names[count] <- paste(names(training_bc)[i], names(training_bc)[j], sep = ".")
    #col_names[count] <- paste("x",i,j,sep=".")
    count <- count + 1
  }
}

# Generate all indices without duplicates (1x8 and 8x1, or 1x1, 2x2, etc)
indices <- matrix(data = NA, nrow = 55000, ncol = 28)

count <- 1
col_names <- character(28)
for (i in 1:8) {
  for (j in i:8) {
    if(i != j) {
      indices[, count] <- nre_fun(training_bc[,i], training_bc[,j]) * 10000
      col_names[count] <- paste(names(training_bc)[i], names(training_bc)[j], sep = ".")
      #col_names[count] <- paste("x",i,j,sep=".")
      count <- count + 1
    }
  }
}

#name columns and create indices_df

colnames(indices) <- col_names
indices_df<-as.data.frame(indices)
indices_df$Classname<-training_bc$Classname
head(indices_df)


###train model
library(caret)
library(ranger)
set.seed(8)
inTrain <- createDataPartition(y = indices_df$Classname,
                               ## the outcome data are needed
                               p = .90,
                               ## The percentage of data in the
                               ## training set
                               list = FALSE)
training <- indices_df[ inTrain,]
testing <- indices_df[-inTrain,]

rf_fit<- ranger(Classname ~ ., mtry = 5,  num.trees = 200, importance="permutation", data = training)

# display results
print(rf_fit)

# load("side_presence_contemp.RData")
rpartPred2 <- predict(rf_fit, testing)
confusionMatrix(rpartPred2$predictions, testing$Classname)
confusion_categories <- confusionMatrix(rpartPred2$predictions, testing$Classname)[["byClass"]]
confusion_DF <- as.data.frame(confusion_categories)
balanced_accuracy <- confusion_categories[,11]

v<-as.vector(rf_fit$variable.importance)
w<-(as.vector((row.names(as.data.frame(rf_fit$variable.importance)))))
DF<-cbind(w,v)
DF<-as.data.frame(DF)
DF$v<-as.numeric(as.character(DF$v))
DF

ggplot(DF, aes(x=reorder(w,v), y=v,fill=v))+ 
  geom_bar(stat="identity", position="dodge")+ coord_flip()+
  ylab("Variable Importance")+
  xlab("")+
  ggtitle("Information Value Summary")+
  scale_fill_gradient(low="red", high="blue")

# data table with colors and variable importance
library(stringr)
sep_col <- str_split_fixed(string = DF$w, pattern = "[.]", n = 2)
DF_sep <- cbind(sep_col, v)
DF_sep <- as.data.frame(DF_sep)
DF_sep <- dplyr::arrange(DF_sep, desc(DF_sep$v))
colnames(DF_sep) <- c("color1", "color2", "v")
write.csv(DF_sep, file = "indices_table_10K_sub_5_indices.csv")
write.csv(balanced_accuracy, file = "class_accuracy_5_indices.csv")
write.csv(rf_fit$prediction.error, file = "rf_fit_5_indices.csv")