function [NMSE]= NMSEfun(y_pred,y_test)
m=mean(y_test);
error = y_test - y_pred; 
SSR = sum(error.^2);
SST = sum((y_test-m).^2);
NMSE=SSR/SST;
end