data=readtable('train.csv');
samplesize=size(data,1);
a=round(samplesize*60/100);
b=round(samplesize*80/100);
data=table2array(data);
%for each col subtract by min to make min=0 and then divide my max-min to scale in [0,1] interval
for i = 1 : size(data,2)
    data(:,i) = (data(:,i)-min(data(:,i)))/(max(data(:,i))-min(data(:,i)));
end

Dtrn=data(1:a,:);
Dval=data(a+1:b,:);
Dchk=data(b+1:end,:);

x_test=Dchk(:,1:81);
y_test=Dchk(:,82);


%{ script to visualize how weights of predictors are affected by k (nearest neighbours amount)
x_train=Dtrn(:,1:81);
y_train=Dtrn(:,82); %number of observations
n=size(x_train,2); %number of predictor variables
weightArray=zeros(6,n);
for k=5:10
    [~,weights] = relieff(x_train,y_train,k);
    weightArray(k,:)=weights;
end
plot(weightArray)%x axis represents rows(no of neighbours) and y axis represents weights of predictors
%}
kopt=6;%we choose as optimal k the point where the lines stop fluctuating



numOfFeatures = [10,15,20]; 
radius = [0.2,0.4,0.8]; %radius of 0.1 returns NaN results in the model
a=size(numOfFeatures,2);
b=size(radius,2);
minError=100;

AllErrors=NaN(a,b);
for i = 1:a
    for j= 1:b
        
        f = numOfFeatures(i);
        ra = radius(j);
        options = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',ra);
        C_errors = zeros(5,1);
        
        for k = 1:5 %5-fold cross validation
            [train,val] = crossValidation(5,data,k);
            
            x_train=train(:,1:81);
            y_train=train(:,82);
            x_val=val(:,1:81);
            y_val=val(:,82);
            
            [idx,~] = relieff(x_train,y_train,kopt); %dim reduction
            
            %keep only f predictor variables for both training and validation data
            train = [x_train(:,idx(1:f)) y_train];
            val = [x_val(:,idx(1:f)) y_val];
            

            fis = genfis(x_train(:,idx(1:f)) ,y_train, options); %generate model
            [trnFis,trnError,stepSize,valFis,~] = anfis(train,fis,50,[],val);
            y_pred = evalfis(x_test(:,idx(1:f)),valFis);
            RMSE = sqrt(mse(y_test, y_pred));
            C_errors(k)=RMSE;
        end           
        AllErrors(i,j) = mean(C_errors);
        if AllErrors(i,j)<minError
            minError=AllErrors(i,j);
            optimalRadius=ra;
            optimalFeatures=f;
        end
    end
end


fprintf('The minimum error among the models is %f\n',minError);
fprintf('Optimal radius for clusters is %f\n',optimalRadius);
fprintf('Optimal amount of predictor variables is %f\n',optimalFeatures);


options = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',optimalRadius);

x_train=Dtrn(:,1:81);
y_train=Dtrn(:,82);

x_val=Dval(:,1:81);
y_val=Dval(:,82);


[idx,weights] = relieff(x_train,y_train,kopt); 
train = [x_train(:,idx(1:optimalFeatures)) y_train];
val = [x_val(:,idx(1:optimalFeatures)) y_val]; 

fis = genfis(x_train(:,idx(1:optimalFeatures)) ,y_train, options); %generate model
[trainFis,trainError,stepSize,valFis,valError] = anfis(train,fis,200,[],val);
            
y_pred = evalfis(x_test(:,idx(1:optimalFeatures)),valFis);
RMSE = sqrt(mse(y_test, y_pred));
NMSE = NMSEfun(y_pred,y_test);
NDEI = sqrt(NMSE);
Rsquare=1-NMSE;
    
%learning curves
figure(1);
plot([trainError valError],'LineWidth',3); 
grid on;
ylabel('Error');
xlabel('Iterations'); 
legend('Training Error','Validation Error');
title("learning curves");
    
%Prediction Error
predictionError = y_test - y_pred;
figure(2);
plot(predictionError,'LineWidth',2); 
grid on;
xlabel('Input');
ylabel('Error');
title(" prediction error");
hold on
plot(y_test)
plot(y_pred)
legend('error','real values','predicted values')



figure(3);
subplot(2,2,1)
plotmf(fis,'input',1);
xlabel('input1')
title('Initial model membership functions')
        
subplot(2,2,2)
plotmf(fis,'input',2);
xlabel('input2')
title('Initial model membership functions')
        
subplot(2,2,3)
plotmf(fis,'input',3);
xlabel('input3')
title('Initial model membership functions')
        
subplot(2,2,4)
plotmf(fis,'input',4);
xlabel('input4')
title('Initial model membership functions')


figure(4);
subplot(2,2,1)
plotmf(valFis,'input',1);
xlabel('input1')
title('Final model membership functions')
        
subplot(2,2,2)
plotmf(valFis,'input',2);
xlabel('input2')
title('Final model membership functions')
        
subplot(2,2,3)
plotmf(valFis,'input',3);
xlabel('input3')
title('Final model membership functions')
        
subplot(2,2,4)
plotmf(valFis,'input',4);
xlabel('input4')
title('Final model membership functions')

fprintf('Rsquare is %f \n',Rsquare);
fprintf('RMSE is %f \n',RMSE);
fprintf('NMSE is %f \n',NMSE);
fprintf('NDEI is %f \n\n',NDEI);




