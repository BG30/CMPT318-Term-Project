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
setwd("C:/Documents/School/6 Spring 2023/CMPT 318/Final Project")
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



