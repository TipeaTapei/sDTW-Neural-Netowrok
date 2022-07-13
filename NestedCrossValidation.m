%%%% NESTED_CROSS_VALIDATION

% The dataset on which we train the network consists of 5 repetitions of
% the power grasp motion (obtained from a simulative model of the surface electromyography
% generation process)

% The nested cross validation is thus made of 5 OUTER LOOPS: in each one a 
% different repetition is used as test set, while the other four repetitions are used for training.
% Each outer loop training set further consists in 4 INNER LOOPS (one repetition at 
% a time is used as validation set, the other three repetitions as train set).

% In each inner loop we train the network 5 times for each set of
% parameters, with different initialization values.

%%%%%%%%%% Load input and target sets %%%%%%%%%%
load('RMS.mat');
input_repetition{1} = RMS{1}(:,1:75:end);
input_repetition{2} = RMS{2}(:,1:75:end);
input_repetition{3} = RMS{3}(:,1:75:end);
input_repetition{4} = RMS{4}(:,1:75:end);
input_repetition{5} = RMS{5}(:,1:75:end);
input_OUTER = [input_repetition{1},input_repetition{2},input_repetition{3},input_repetition{4},input_repetition{5}];

load('ref.mat');
target_repetition{1} = ref{1}(:,1:75:end);
target_repetition{2} = ref{2}(:,1:75:end);
target_repetition{3} = ref{3}(:,1:75:end);
target_repetition{4} = ref{4}(:,1:75:end);
target_repetition{5} = ref{5}(:,1:75:end);
target_OUTER = [target_repetition{1},target_repetition{2},target_repetition{3},target_repetition{4},target_repetition{5}];

cvFoldsOUTER = [ones(1,length(input_repetition{1})),2*ones(1,length(input_repetition{2})),3*ones(1,length(input_repetition{3})),4*ones(1,length(input_repetition{4})),5*ones(1,length(input_repetition{5}))];

%%%%%%%%%% Parameters we want to test %%%%%%%%%%
parameters = {[4,4];[6,6];[8,8];[10,10]};

%%%  (5,4,4,5) = ( OUTER_ITER , INNER_ITER , SET_OF_PARAM , REPETITIONS )

netArray = cell(5,4,4,5);      % Stores all trained networks

performanceVAL_DTW = zeros(5,4,4,5);      % Stores the performances on the validation set (using DTW measure)
performanceTRAIN_DTW = zeros(5,4,4,5);    % Stores the performances on the train set (using DTW measure)

INITIALIZATION = 1;      % 1:initialize weights and biases   0:DEFAULT initialization

% Change according to the desired target
ONE_THIRD = 1;           % 1:stretch the signal to 1/3    0:DO NOT stretch 
TWO_THIRD = 0;           % 1:stretch the signal to 2/3    0:DO NOT stretch 
UNSTRETCHED = 0;

for ii=1:5    % OUTER_ITERATIONS
    
    %%%%%%%%%% Separate training from test data of OUTER LOOP %%%%%%%%%%
    testIdx = (cvFoldsOUTER == ii);
    trainingIdx =~ testIdx;
    
    xTest = input_OUTER(:,testIdx);
    xTraining = input_OUTER(:,trainingIdx);
    tTestFIX = target_OUTER(:,testIdx);
    tTraining = target_OUTER(:,trainingIdx);
    
    h = 1;
    cvFoldsINNER = [];
    input_INNER = [];
    target_INNER = [];
    
    for jj=1:5
        if jj ~= ii
            cvFoldsINNER = [cvFoldsINNER, h*ones(1,length(input_repetition{jj}))];
            input_INNER = [input_INNER, input_repetition{jj}];
            target_INNER= [target_INNER, target_repetition{jj}];
            h = h+1;
        end
    end
    
    for kk=1:4     % INNER_ITERATION
            
        %%%%%%%%%%% Get train and test data of INNER LOOP %%%%%%%%%%%
        validationIdx = (cvFoldsINNER == kk);
        trainIdx =~ validationIdx;
        
        xValidation = input_INNER(:,validationIdx);
        xTrain = input_INNER(:,trainIdx);
        tValidationFIX = target_INNER(:,validationIdx);
        tTrainFIX = target_INNER(:,trainIdx);
        
        %%%%%%%%%% Stretch the target signal if requested %%%%%%%%%%
        if ONE_THIRD
            tempTrain = interp1(tTrainFIX,1:3:length(tTrainFIX));
            tTrain = [tempTrain,ones(1,length(tTrainFIX)-length(tempTrain))];
        end
        
        if TWO_THIRD
            tempTrain = interp1(tTrainFIX,1:1.5:length(tTrainFIX));
            tTrain = [tempTrain,ones(1,length(tTrainFIX)-length(tempTrain))];
        end
        
        if UNSTRETCHED
            tTrain = tTrainFIX;
        end
        
        for ll=1:4     % PARAMETERS_ITERATIONS
            
            for mm=1:5     % REPETITIONS
             
                %%%%%%%%%%% Create the network and set the parameters %%%%%%%%%%%
                net = fitnet(parameters{ll},'trainrp');
                net = configure(net,xTrain,tTrain);

                net.trainParam.epochs = 400;
                net.divideParam.trainRatio = 1;
                net.divideParam.testRatio = 0;
                net.divideParam.valRatio = 0;
                net.output.processFcns = {'mapminmax'};

                %%%%%%%%%%% Set the performance function %%%%%%%%%%%
                net.performFcn = 'myperformance';

                %%%%%%%%%%% Initialize weights and biases %%%%%%%%%%%
                if INITIALIZATION 
                    net.initFcn = 'initlay';
                    for nn=1:size(net.layers,1)
                        net.layers{nn}.initFcn = 'initwb';
                    end

                    initialWeightsFunction = 'rands';
                    initialBiasesFunction = 'rands';

                    for nn=1:size(net.inputWeights,1)
                        for pp=1:size(net.inputWeights,2)
                            net.inputWeights{nn,pp}.initFcn = initialWeightsFunction;
                        end
                    end

                    for nn=1:size(net.layerWeights,1)
                        for pp=1:size(net.layerWeights,2)
                            net.layerWeights{nn,pp}.initFcn = initialWeightsFunction;
                        end
                    end    

                    for nn=1:size(net.biases,1)
                        for pp=1:size(net.biases,2)
                            net.biases{nn,pp}.initFcn = initialBiasesFunction;
                        end
                    end   

                    net=init(net);
                end

                %%%%%%%%%%% Train the network %%%%%%%%%%%
                [net,~] = train(net,xTrain,tTrain);

                %%%%%%%%%%% Evaluate the performances %%%%%%%%%%%
                yTrain = net(xTrain);
                yValidation = net(xValidation);
                performanceTRAIN_DTW(ii,kk,ll,mm) = dtw(yTrain,tTrainFIX);
                performanceVAL_DTW(ii,kk,ll,mm) = dtw(yValidation,tValidationFIX);

                %%%%%%%%%%% Store the results %%%%%%%%%%%
                netArray{ii,kk,ll,mm} = net;
            end
        end
    end
end


%%%%%%%%%% For each parameter set compute mean performances %%%%%%%%%%
meanPerfDTWArray = zeros(5,4);    

for ii=1:5
    for ll=1:4
        meanPerfDTWArray(ii,ll) = mean(mean(performanceVAL_DTW(ii,:,ll,:)));
    end
end

%%%%%%%%%% Select the best set of parameters for each outer loop %%%%%%%%%%
BestNetSETid = zeros(5,1);
BestNet = cell(5,1);

for ii=1:5
    [~,BestNetSETid(ii)] = min(meanPerfDTWArray(ii,:));
end

%%%%%%%%%% Evaluate the best model %%%%%%%%%%

for ii=1:5    % OUTER_ITERATIONS
    
    %%%%%%%%%% Separate training from test data %%%%%%%%%%
    testIdx = (cvFoldsOUTER == ii);
    trainingIdx =~ testIdx;
    
    xTest = input_OUTER(:,testIdx);
    xTraining = input_OUTER(:,trainingIdx);
    tTestFIX = target_OUTER(:,testIdx);
    tTrainingFIX = target_OUTER(:,trainingIdx);
        
    if ONE_THIRD
        tempTraining = interp1(tTrainingFIX,1:3:length(tTrainingFIX));
        tTraining = [tempTraining,ones(1,length(tTrainingFIX)-length(tempTraining))];
    end

    if TWO_THIRD
        tempTraining = interp1(tTrainingFIX,1:1.5:length(tTrainingFIX));
        tTraining = [tempTraining,ones(1,length(tTrainingFIX)-length(tempTraining))];
    end

    if UNSTRETCHED
        tTraining = tTrainingFIX;
    end
                    
    %%%%%%%%%%% Create the network and set the parameters %%%%%%%%%%%
    net = fitnet(parameters{BestNetSETid(ii)},'trainrp');
    net = configure(net,xTraining,tTraining);

    net.trainParam.epochs = 400;
    net.divideParam.trainRatio = 1;
    net.divideParam.testRatio = 0;
    net.divideParam.valRatio = 0;
    net.output.processFcns = {'mapminmax'};

    %%%%%%%%%%% Set the performance function %%%%%%%%%%%
    net.performFcn = 'myperformance';

    %%%%%%%%%%% Initialize weights and biases %%%%%%%%%%%
    if INITIALIZATION 
        net.initFcn = 'initlay';
        for nn=1:size(net.layers,1)
            net.layers{nn}.initFcn = 'initwb';
        end

        initialWeightsFunction = 'rands';
        initialBiasesFunction = 'rands';

        for nn=1:size(net.inputWeights,1)
            for pp=1:size(net.inputWeights,2)
                net.inputWeights{nn,pp}.initFcn = initialWeightsFunction;
            end
        end

        for nn=1:size(net.layerWeights,1)
            for pp=1:size(net.layerWeights,2)
                net.layerWeights{nn,pp}.initFcn = initialWeightsFunction;
            end
        end    

        for nn=1:size(net.biases,1)
            for pp=1:size(net.biases,2)
                net.biases{nn,pp}.initFcn = initialBiasesFunction;
            end
        end   

        net=init(net);
    end

    %%%%%%%%%%% Train the network %%%%%%%%%%%
    [net,~] = train(net,xTraining,tTraining);

    %%%%%%%%%%% Evaluate the performances on the test set %%%%%%%%%%%
    yTraining = net(xTraining);
    yTest = net(xTest);
    performanceTRAINING_DTW(ii) = dtw(yTraining,tTrainingFIX);
    performanceTEST_DTW(ii) = dtw(yTest,tTestFIX);
    figure; plot(yTest); hold on; plot(tTestFIX); legend('Test Output','Synchronized signal');

    %%%%%%%%%%% Store net, training and test sets %%%%%%%%%%%
    BestNet{ii,1} = net;
    TrainingSet{ii,1} = xTraining;
    TrainingTGSet{ii,1} = tTrainingFIX;
    TrainingTGSet_stretc{ii,1} = tTraining;
    TestSet{ii,1} = xTest;
    TestTGSet{ii,1} = tTestFIX;
end
