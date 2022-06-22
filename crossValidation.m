function [train,val] = crossValidation(k,data,i)
    sz=size(data,1);
    v=round(sz/k); %percentage used as validation data
    floor=((i-1)*v)+1;
    ceil=i*v;
    if ceil>sz %avoid exceeding bounds
        ceil=sz;
    end
    val=data(floor:ceil,:);
    if i==1
        train =data(ceil+1:sz,:); 
    elseif i==k
        train =data(1:floor-1,:); 
    else
    train =[data(1:floor-1,:); data(ceil+1:sz,:)]; 
    end
end
  

