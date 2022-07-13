% Plot the results of the nested cross validation: for each OUTER loop we
% plot the output of the softDTW network for training and test sets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('ws_cross_validation.mat') % comment this line if you create your own
                                % w.s. using NestedCrossValidation.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ii=1:5    % OUTER_ITERATIONS
    
    net = BestNet{ii};

    yTraining = net(TrainingSet{ii,1});
    yTest = net(TestSet{ii,1});
    
    figure; 
    subplot(2,1,1); plot(yTraining,'LineWidth',1.5); hold on; plot(TrainingTGSet{ii,1},'LineWidth',1.5); legend('Training Output','Synchronized Signal');
    subplot(2,1,2); plot(yTest,'LineWidth',1.5); hold on; plot(TestTGSet{ii,1},'LineWidth',1.5); legend('Test Output','Synchronized Signal');
    
end