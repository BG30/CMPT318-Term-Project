#library(devtools)
#install_github("vqv/ggbiplot")
require(ggbiplot)
library(dplyr)
library(lubridate)
library(ggplot2)
library(depmixS4) 
library(depmixS4)
library(data.table)
library(hash)

### Format time ###
format_time <- function(data) {
  startTime <- strptime("06:20:00", format="%H:%M:%S")
  endTime <- strptime("10:20:00", format="%H:%M:%S")
  data <- subset(data, wday(as.Date(data$Date, format = "%d/%m/%y")) == 5)
  data$Date <- as.Date(data$Date, format = "%d/%m/%y")
  startTime <- strptime("06:20:00", format="%H:%M:%S")
  endTime <- strptime("10:20:00", format="%H:%M:%S")
  data <- subset(data, difftime(strptime(data$Time, format="%H:%M:%S"), startTime) >= 0 & 
                          difftime(strptime(data$Time, format="%H:%M:%S"), endTime) < 0)
  
  # removing some data for dates that do not meet the threshold of having 720 entries per date
  data <- subset(data, !(Date == "2020-08-06"))
  data <- subset(data, !(Date == "2020-08-13"))
  data <- subset(data, !(Date == "2020-04-30"))
  data <- subset(data, !(Date == "2020-12-03"))
  data <- subset(data, !(Date == "2020-12-10"))
  
  return(data)
}

### Import data ###
setwd("C:/Documents/School/6 Spring 2023/CMPT 318/Final Project")
df <- read.table("Term_Project_Dataset.txt", header = TRUE, sep = ",")
### df <- na.omit(df)

### Scale data ###
scaled_data <- cbind(df["Date"], df["Time"], scale(df[, c(3:9)]))

pcs <- prcomp(scaled_data[, c(3:9)])
### Filter weekday and timeframe ###
scaled_data <- format_time(scaled_data)

### Create sample matrix ###
scaled_data$week_num <- strtoi(strftime(scaled_data$Date, format = "%V"))
scaled_data <- group_by(scaled_data, week_num)

# Average of each feature by week
weekly_samples_Global_active_power <- summarise(scaled_data, Global_active_power = mean(Global_active_power))
weekly_samples_Global_reactive_power <- summarise(scaled_data, Global_reactive_power = mean(Global_reactive_power))
weekly_samples_Global_Voltage <- summarise(scaled_data, Voltage = mean(Voltage))
weekly_samples_Global_intensity <- summarise(scaled_data, Global_intensity = mean(Global_intensity))
weekly_samples_Sub_metering_1 <- summarise(scaled_data, Sub_metering_1 = mean(Sub_metering_1))
weekly_samples_Sub_metering_2 <- summarise(scaled_data, Sub_metering_2 = mean(Sub_metering_2))
weekly_samples_Sub_metering_3 <- summarise(scaled_data, Sub_metering_3 = mean(Sub_metering_3))

# Create a vector of data frames
weekly_samples <- list(weekly_samples_Global_active_power, 
                       weekly_samples_Global_reactive_power, 
                       weekly_samples_Global_Voltage,
                       weekly_samples_Global_intensity,
                       weekly_samples_Sub_metering_1,
                       weekly_samples_Sub_metering_2,
                       weekly_samples_Sub_metering_3)

# Samples of the average output per feature by week
weekly_samples <- Reduce(function(x, y) merge(x, y, by = "week_num", all.x = TRUE), weekly_samples)
weekly_samples = weekly_samples[,-1]

print(weekly_samples)

# PCA Analysis
pcs <- prcomp(weekly_samples)

plot(pcs)
summary(pcs)
print(pcs)

ggbiplot(pcs)


################    HMM Training and Testing   ############
scaled_data <- cbind(df["Date"], df["Time"], scale(df[, c(3:9)]))
data <- format_time(scaled_data)


# Split into train and test data
train_dates <- unique(data$Date)[1:38]
test_dates <- unique(data$Date)[39:48]
train_data <- subset(data, Date %in% train_dates)
test_data <- subset(data, Date %in% test_dates)

count_vector_train <- train_data %>% 
  group_by(Date) %>% 
  summarize(count = n()) %>%
  pull(count)
count_vector_test <- test_data %>% 
  group_by(Date) %>% 
  summarize(count = n()) %>%
  pull(count)


train_data <- train_data[,c("Global_active_power", "Voltage", "Sub_metering_3")]
test_data <- test_data[,c("Global_active_power", "Voltage", "Sub_metering_3")]
test_data <- test_data[is.finite(rowSums(test_data)),]
train_data <- train_data[is.finite(rowSums(train_data)),]


# Training and testing the HMMs
set.seed(4)
res_list <- array(c(0), dim = c(21,4))


for (i in 4:23) {
  print(paste("nstates ", i, " run ", i-3))
  
  modFit <- depmix(response=list(train_data$Global_active_power ~ 1, train_data$Voltage ~ 1, train_data$Sub_metering_3 ~ 1), 
                   data = train_data, nstates = i, 
                   ntimes = count_vector_train,
                   family=list(gaussian(), gaussian(), gaussian())
  )
  
  fm <- fit(modFit)
  
  
  modTest <- depmix(response=list(test_data$Global_active_power ~ 1, test_data$Voltage ~ 1, test_data$Sub_metering_3 ~ 1), 
                    data = test_data, nstates = i, 
                    ntimes = count_vector_test,
                    family=list(gaussian(), gaussian(), gaussian())
  )
  modTest <- setpars(modTest, getpars(fm))
  fb <- forwardbackward(modTest)
  fb$logLike
  
  # Loglik normailization
  modelLogLike <- logLik(fm)
  testLogLik <- fb$logLike
  norm_modelLogLike <- modelLogLike/nrow(train_data)
  norm_testLogLik <- testLogLik/nrow(test_data)
  
  res_list[i-2, 1] = norm_modelLogLike
  res_list[i-2, 2] = AIC(fm)
  res_list[i-2, 3] = BIC(fm)
  res_list[i-2, 4] = norm_testLogLik
  print(paste("logLik ", norm_modelLogLike, " AIC ", AIC(fm), " BIC ", BIC(fm), " Test logLik ", norm_testLogLik))
}


print(res_list)

d <- data.frame(res_list)
names(d) = c("logLik", "AIC", "BIC", "Test LogLik")
d$nstates = c(0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
d <- d[-1,]
print(d)

plot(unlist(d[1]), unlist(d[3]), xlab = "logLik", ylab = "BIC")
plot(unlist(d[5]), unlist(d[1]), xlab = "nstates", ylab = "logLik")
plot(unlist(d[5]), unlist(d[3]), xlab = "nstates", ylab = "BIC")
plot(unlist(d[5]), unlist(d[2]), xlab = "nstates", ylab = "AIC")
plot(unlist(d[5]), unlist(d[4]), xlab = "nstates", ylab = "Test-LogLik")




################    Anomaly Detection   ############
# rebuild the model based on the best training model
# will train it on both test and training data
chosen_states = 7
train_data <- cbind(df["Date"], df["Time"], scale(df[, c(3:9)]))
train_data <- format_time(scaled_data)
train_dates <- unique(data$Date)[1:38]
train_data <- subset(data, Date %in% train_dates)

count_vector_train <- train_data %>% 
  group_by(Date) %>% 
  summarize(count = n()) %>%
  pull(count)

finalModel <- depmix(response=list(train_data$Global_active_power ~ 1, train_data$Voltage ~ 1, train_data$Sub_metering_3 ~ 1), 
                 data = train_data, nstates = chosen_states, 
                 ntimes = count_vector_train,
                 family=list(gaussian(), gaussian(), gaussian())
)
fm <- fit(finalModel)
logLik(fm)

setwd("C:/Documents/School/6 Spring 2023/CMPT 318/Final Project/Data_with_Anomalies")
dataset1 <- read.table("Dataset_with_Anomalies_1.txt", header = TRUE, sep = ",")
dataset2 <- read.table("Dataset_with_Anomalies_2.txt", header = TRUE, sep = ",")
dataset3 <- read.table("Dataset_with_Anomalies_3.txt", header = TRUE, sep = ",")
dataset1 <- na.omit(dataset1)
dataset2 <- na.omit(dataset2)
dataset3 <- na.omit(dataset3)
dataset1 <- cbind(dataset1["Date"], dataset1["Time"], scale(dataset1[, c("Global_active_power", "Voltage", "Sub_metering_3")]))
dataset2 <- cbind(dataset2["Date"], dataset2["Time"], scale(dataset2[, c("Global_active_power", "Voltage", "Sub_metering_3")]))
dataset3 <- cbind(dataset3["Date"], dataset3["Time"], scale(dataset3[, c("Global_active_power", "Voltage", "Sub_metering_3")]))
dataset1 <- format_time(dataset1)
dataset2 <- format_time(dataset2)
dataset3 <- format_time(dataset3)

## Dataset 1
count_vector <- dataset1 %>% 
  group_by(Date) %>% 
  summarize(count = n()) %>%
  pull(count)

modTest <- depmix(response=list(dataset1$Global_active_power ~ 1, dataset1$Voltage ~ 1, dataset1$Sub_metering_3 ~ 1), 
                  data = dataset1, nstates = chosen_states, 
                  ntimes = count_vector,
                  family=list(gaussian(), gaussian(),  gaussian())
)
modTest <- setpars(modTest, getpars(fm))
fb <- forwardbackward(modTest)

# Loglik normailization
modelLogLike <- logLik(fm)
testLogLik <- fb$logLike
norm_modelLogLike <- modelLogLike/nrow(data)
norm_testLogLik <- testLogLik/nrow(dataset1)
print(paste("logLik of model: ", norm_modelLogLike, "logLik on dataset1:", norm_testLogLik))


## DataSet 2
count_vector <- dataset2 %>% 
  group_by(Date) %>% 
  summarize(count = n()) %>%
  pull(count)


modTest <- depmix(response=list(dataset2$Global_active_power ~ 1, dataset2$Voltage ~ 1, dataset2$Sub_metering_3 ~ 1), 
                  data = dataset2, nstates = chosen_states, 
                  ntimes = count_vector,
                  family=list(gaussian(), gaussian(), gaussian())
)
modTest <- setpars(modTest, getpars(fm))
fb <- forwardbackward(modTest)

# Loglik normailization
modelLogLike <- logLik(fm)
testLogLik <- fb$logLike
norm_modelLogLike <- modelLogLike/nrow(data)
norm_testLogLik <- testLogLik/nrow(dataset2)
print(paste("logLik of model: ", norm_modelLogLike, "logLik on dataset2:", norm_testLogLik))



## DataSet 3
count_vector <- dataset3 %>% 
  group_by(Date) %>% 
  summarize(count = n()) %>%
  pull(count)


modTest <- depmix(response=list(dataset3$Global_active_power ~ 1, dataset3$Voltage ~ 1, dataset3$Sub_metering_3 ~ 1), 
                  data = dataset3, nstates = chosen_states, 
                  ntimes = count_vector,
                  family=list(gaussian(), gaussian(), gaussian())
)
modTest <- setpars(modTest, getpars(fm))
fb <- forwardbackward(modTest)

# Loglik normailization
modelLogLike <- logLik(fm)
testLogLik <- fb$logLike
norm_modelLogLike <- modelLogLike/nrow(data)
norm_testLogLik <- testLogLik/nrow(dataset3)
print(paste("logLik of model: ", norm_modelLogLike, "logLik on dataset3:", norm_testLogLik))
