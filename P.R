install.packages("devtools")
library(devtools)
install_github("vqv/ggbiplot")
require(ggbiplot)
library(dplyr)
library(lubridate)
library(ggplot2)
library(depmixS4) 

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

# Split into train and test data
# have 53 days, training will have 37 days and testing will have 16 days
train_dates <- unique(data$Date)[1:37]
test_dates <- unique(data$Date)[38:53]
train_data <- subset(data, Date %in% train_dates)
test_data <- subset(data, Date %in% test_dates)


sum(train_data$Date == "2020-08-06")
length(unique(data$Date))
unique(train_data$Date)


# Training the HMM
set.seed(1)
for (i in 4:24) {
  print(paste("nstates ", i, " run ", i-2))
  
  mod <- depmix(response =list(train_data$Global_active_power ~ 1, train_data$Voltage ~ 1, train_data$Sub_metering_3 ~ 1), 
                data = train_data, nstates = i, 
                ntimes = rep(c(720), each=37)
                )
  
  fm <- fit(mod)
  
  res_list[i-2, 1] = logLik(fm)
  res_list[i-2, 2] = AIC(fm)
  res_list[i-2, 3] = BIC(fm)
  print(paste("logLik ", logLik(fm), " AIC ", AIC(fm), " BIC ", BIC(fm)))
}


print(res_list)

d <- data.frame(res_list)
names(d) = c("logLik", "AIC", "BIC")
d$nstates = seq(from = 4, to = 24)
print(d)

plot(unlist(d[1]), unlist(d[3]), xlab = "logLik", ylab = "BIC")
plot(unlist(d[4]), unlist(d[1]), xlab = "nstates", ylab = "logLik")
plot(unlist(d[4]), unlist(d[3]), xlab = "nstates", ylab = "BIC")


# Testing the HMM





################    Anomaly Detection   ############