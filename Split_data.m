x=load('airfoil_self_noise.dat');
samplesize=size(x,1);
a=round(samplesize*60/100);
b=round(samplesize*80/100);

Dtrn=x(1:a,:);
Dval=x(a+1:b,:);
Dchk=x(b+1:end,:);

x_train=Dtrn(:,1:5);
y_train=Dtrn(:,6);

x_val=Dval(:,1:5);
y_val=Dval(:,6);

x_test=Dchk(:,1:5);
y_test=Dchk(:,6);