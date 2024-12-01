#importing datasaet
dataset = read.csv('~/machine_learning_udemy/data.csv')
#missing data
dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), dataset$Age)