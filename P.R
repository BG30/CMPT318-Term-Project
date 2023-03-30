install.packages("devtools")
library(devtools)
install_github("vqv/ggbiplot")
require(ggbiplot)
library(dplyr)
library(lubridate)
library(ggplot2)
library(depmixS4)
library(data.table)
library(hash)

### Import data ###
getwd()
setwd("C:/Users/ricks/Desktop/CMPT 318/FinalProject/Data")
df <- read.table("Term_Project_Dataset.txt", header = TRUE, sep = ",")
df <- na.omit(df)

### Scale data ###
scaled_data <- cbind(df["Date"], df["Time"], scale(df[, c(3:9)]))

### Filter weekday and timeframe ###
scaled_data <- subset(scaled_data, wday(as.Date(scaled_data$Date, format = "%d/%m/%y")) == 5)
scaled_data$Date <- as.Date(scaled_data$Date, format = "%d/%m/%y")
startTime <- strptime("06:20:00", format="%H:%M:%S")
endTime <- strptime("10:20:00", format="%H:%M:%S")
scaled_data <- subset(scaled_data, difftime(strptime(scaled_data$Time, format="%H:%M:%S"), startTime) >= 0 & 
                        difftime(strptime(scaled_data$Time, format="%H:%M:%S"), endTime) < 0)

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
scaled_data <- subset(scaled_data, wday(as.Date(scaled_data$Date, format = "%d/%m/%y")) == 5)
scaled_data$Date <- as.Date(scaled_data$Date, format = "%d/%m/%y")
startTime <- strptime("06:20:00", format="%H:%M:%S")
endTime <- strptime("10:20:00", format="%H:%M:%S")
scaled_data <- subset(scaled_data, difftime(strptime(scaled_data$Time, format="%H:%M:%S"), startTime) >= 0 & 
                        difftime(strptime(scaled_data$Time, format="%H:%M:%S"), endTime) < 0)
data <- scaled_data[,c("Date", "Time", "Global_active_power", "Voltage", "Sub_metering_3")]
# removing some data for dates that do not meet the threshold of having 720 entries per date
data <- subset(data, !(Date == "2020-08-06"))
data <- subset(data, !(Date == "2020-08-13"))
data <- subset(data, !(Date == "2020-04-30"))
data <- subset(data, !(Date == "2020-12-03"))
data <- subset(data, !(Date == "2020-12-10"))


# Split into train and test data
train_dates <- unique(data$Date)[1:24]
test_dates <- unique(data$Date)[25:48]
train_data <- subset(data, Date %in% train_dates)
test_data <- subset(data, Date %in% test_dates)
train_data <- train_data[,c("Global_active_power", "Voltage", "Sub_metering_3")]
test_data <- test_data[,c("Global_active_power", "Voltage", "Sub_metering_3")]
test_data <- test_data[is.finite(rowSums(test_data)),]
train_data <- train_data[is.finite(rowSums(train_data)),]


# Training and testing the HMMs
set.seed(3)
res_list <- array(c(0), dim = c(22,4))


for (i in 4:24) {
  print(paste("nstates ", i, " run ", i-2))
  
  modFit <- depmix(response=list(train_data$Global_active_power ~ 1, train_data$Voltage ~ 1, train_data$Sub_metering_3 ~ 1), 
                data = train_data, nstates = 3, 
                ntimes = rep(c(720), each=24),
                family=list(gaussian(), gaussian(), gaussian())
         )
  fm <- fit(modFit, em=em.control((maxit = 900000)))
  
  
  modTest <- depmix(response=list(train_data$Global_active_power ~ 1, train_data$Voltage ~ 1, train_data$Sub_metering_3 ~ 1), 
                    data = test_data, nstates = 3, 
                    ntimes = rep(c(720), each=24),
                    family=list(gaussian(), gaussian(), gaussian())
             )
  modTest <- setpars(modTest , getpars(modTest))
  fb <- forwardbackward(modTest)
  fb$logLike
  logLik(modTest)
  
  res_list[i-2, 1] = logLik(fm)
  res_list[i-2, 2] = AIC(fm)
  res_list[i-2, 3] = BIC(fm)
  res_list[i-2, 4] = logLik(modTest)
  print(paste("logLik ", logLik(fm), " AIC ", AIC(fm), " BIC ", BIC(fm), " Test logLik ", fb$logLike))
}


print(res_list)

d <- data.frame(res_list)
names(d) = c("logLik", "AIC", "BIC", "Test LogLik")
d$nstates = c(0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)
d <- d[-1,]
print(d)

plot(unlist(d[1]), unlist(d[3]), xlab = "logLik", ylab = "BIC")
plot(unlist(d[5]), unlist(d[1]), xlab = "nstates", ylab = "logLik")
plot(unlist(d[5]), unlist(d[3]), xlab = "nstates", ylab = "BIC")
plot(unlist(d[5]), unlist(d[2]), xlab = "nstates", ylab = "AIC")
plot(unlist(d[5]), unlist(d[4]), xlab = "nstates", ylab = "Test=LogLik")




################    Anomaly Detection   ############
# rebuild the model based on the best training model
mod <- depmix(response=list(train_data$Global_active_power ~ 1, train_data$Voltage ~ 1, train_data$Sub_metering_3 ~ 1), 
              data = train_data, nstates = i, 
              ntimes = rep(c(720), each=40),
              family=list(gaussian(), gaussian(), gaussian())
)

fm <- fit(mod)

setwd("C:/Users/ricks/Desktop/CMPT 318/FinalProject/Data/Data_with_Anomalies")
dataset1 <- read.table("Dataset_with_Anomalies_1.txt", header = TRUE, sep = ",")
dataset2 <- read.table("Dataset_with_Anomalies_2.txt", header = TRUE, sep = ",")
dataset3 <- read.table("Dataset_with_Anomalies_3.txt", header = TRUE, sep = ",")
