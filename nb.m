data_country = readtable('data.csv') %reading data

data_missing = ismissing(data_country) %detecting missing values

data_fix = rmmissing(data_country) %removing missing values

[p,q] = size(data_fix); %defining table size
percentage = 0.70; %defining split percentage
index = randperm(p); %making data index
data_train = data_fix(index(1:round(percentage*p)),:) %splitting data into train
testing = data_fix(index(round(percentage*p)+1:end),:) %splitting data into test

xtrain_raw = data_train(:,[2,3,6,7,8]) %For features train variable
ytrain = data_train(:,10) %For class train variable
xtest_raw = testing(:,[2,3,6,7,8]) %For feature test variable
ytest = testing(:,10) %For class test variable

outlier1 = isoutlier(xtrain_raw, 'quartiles') %detecting outlier using IQR
xtrain_wo = filloutliers(xtrain_raw, 'nearest') %filling outlier with the nearest value
outlier2 = isoutlier(xtest_raw, 'quartiles') %detecting outlier using IQR
xtest_wo = filloutliers(xtest_raw, 'nearest') %filling outlier with the nearest value

xtrain = normalize(xtrain_wo,'scale') %normalizing xtrain using scale method
xtest = normalize(xtest_wo, 'scale') %normalizing xtest using scale method

model=fitcnb(xtrain,ytrain,'Distribution','normal'); %train naive bayes model
pdc=predict(model,xtest); %predicting xtest class

fig = figure; %defining a figure
confusion_matrix=confusionchart(table2array(ytest),pdc); %making a confusion chart
confusion_matrix.Title = 'Confusion Matrix'; %giving confusion chart a title
fig_Position = fig.Position; %setting up figure position
fig_Position(3) = fig_Position(3)*1.5; %setting up figure position
fig.Position = fig_Position; %setting up figure position

confusion_matrix_model = confusionmat(table2array(ytest),pdc); %defining a confusion matrix
akurasi = sum(diag(confusion_matrix_model))/sum(confusion_matrix_model(:)) * 100 %calculating accuracy from confusion matrix