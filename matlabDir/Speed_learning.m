clear; clc; close all;

timeKSVDD = zeros(1,1);
timeMODD = zeros(1,1);
for ix=1:2
    
    data0= load('100m.mat');  %contains val: [2x650000 double]
    chunk = 49920;  % DEFAULT: 49920
    train = data0.val(1,1:chunk*ix);
    
    l=chunk*ix/128;  %Number of trained signal
    n=128;  %length of signal
    Data = zeros(n,l);
    
    TrainMat = zeros(n,l);
    for i=1:l
        TrainMat(:,i) = train((i-1)*n+1:i*n);
        %plot(Data(:,i));pause(1)
    end
    
    %-------------dictionary learning
    param.K = 2*n;% number of atom in dictionary
    param.L = 1;
    param.numIteration = 10;
    param.errorFlag = 0;
    param.preserveDCAtom =0;
    param.displayProgress = 0;
    param.InitializationMethod = 'DataElements';
    param.TrueDictionary = randn(n,2*n);
    %param.InitializationMethod = 'DataElements';
    %param.InitializationMethod = 'GivenMatrix';
    
    iniMat = randn(n,param.K);
    for i =1: param.K
        iniMat(:,i) = iniMat(:,i)/norm(iniMat(:,i));%normalizing columns of matrix
    end
    param.initialDictionary = iniMat;
    
    itr=10;
    tic
    for i=1:itr
        [DicMod, outputMod] = MOD(TrainMat,param);
    end
    timeKSVD=toc;
    %save('Sparsifying_ECG_128_256','DicMod');
    
    tic
    for i=1:itr
        [DicKSVD,X] = KSVD(TrainMat,param);
    end
    timeMOD=toc;
    
    timeKSVDD(ix) = timeKSVD /itr;
    timeMODD(ix) = timeMOD/itr;
    
end

disp('KSVD');
disp(timeKSVDD);
disp('MOD');
disp(timeMODD);

% title
title('Time comparison between MOD and KSVD');
% x-axis label
xlabel('Number of trained signal');
% y-axis label
ylabel('Time (s)');
plot(timeKSVDD);
hold on;
plot(timeMODD,'r')
legend('KSVD','MOD');
