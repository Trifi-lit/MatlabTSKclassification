data=load('airfoil_self_noise.dat');
samplesize=size(data,1);
a=round(samplesize*60/100);
b=round(samplesize*80/100);

%for each col subtract by min to make min=0 and then divide my max-min to scale in [0,1] interval
for i = 1 : size(data,2)
    data(:,i) = (data(:,i)-min(data(:,i)))/(max(data(:,i))-min(data(:,i)));
end

Dtrn=data(1:a,:);
Dval=data(a+1:b,:);
Dchk=data(b+1:end,:);

x_train=Dtrn(:,1:5);
y_train=Dtrn(:,6);

x_val=Dval(:,1:5);
y_val=Dval(:,6);

x_test=Dchk(:,1:5);
y_test=Dchk(:,6);


opt1 = genfisOptions('GridPartition');
opt1.OutputMembershipFunctionType = 'constant';
opt2 = genfisOptions('GridPartition');
opt2.OutputMembershipFunctionType = 'constant';
opt2.NumMembershipFunctions = 3; 
opt3 = genfisOptions('GridPartition');
opt4 = genfisOptions('GridPartition');
opt4.NumMembershipFunctions = 3;
%inline option specification did not work in my matlab version
%linear output, gbellmf input and 2 membership functions are the default

fis1 = genfis(x_train,y_train,opt1);
fis2 = genfis(x_train,y_train,opt2);
fis3 = genfis(x_train,y_train,opt3);
fis4 = genfis(x_train,y_train,opt4);
fis=[fis1,fis2,fis3,fis4];

A=NaN(4,4); %store the 4 eval
for i = 1:4
    [trainFis,trainError,~,valFis,valError] = anfis(Dtrn,fis(i),100,[],Dval);
    %trainFis: smallest training error. valFis: smallest validation error
    fis(i)=valFis; %store the new optimal models
    
    y_pred = evalfis(x_test,valFis);
    RMSE = sqrt(mse(y_test, y_pred));
    NMSE = NMSEfun(y_pred,y_test);
    NDEI = sqrt(NMSE);
    Rsquare=1-NMSE;
    A(i,:) = [Rsquare; RMSE; NMSE; NDEI];
    
    %learning curves
    figure(i);
    plot([trainError valError],'LineWidth',3); 
    grid on;
    ylabel('Error');
    xlabel('Iterations'); 
    legend('Training Error','Validation Error');
    str = "TSK model ";
    str = strcat(str,string(i));
    str = strcat(str," learning curve");
    title(str);
    
    
    %Prediction Error
    predictionError = y_test - y_pred;
    figure(4+i);
    plot(predictionError,'LineWidth',2); 
    grid on;
    str = "TSK model ";
    str = strcat(str,string(i));
    str = strcat(str," prediction error");
    xlabel('Input');
    ylabel('Error');
    title(str);
end

for i=1:4
    fprintf('Rsquare for model %d is %f \n',i,A(i,1));
    fprintf('RMSE for model %d is %f \n',i,A(i,2));
    fprintf('NMSE for model %d is %f \n',i,A(i,3));
    fprintf('NDEI for model %d is %f \n\n',i,A(i,4));
end











