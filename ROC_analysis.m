function [AUC] = ROC_analysis(labels,scores,Plot_Flag,Method_Name)
Class_Index = unique(labels);
[ROC_X,ROC_Y,~,AUC] = perfcurve(labels,scores,Class_Index(1));
if AUC<0.5
    [ROC_X,ROC_Y,~,AUC] = perfcurve(labels,scores,Class_Index(2));
end
if nargin<3
    Plot_Flag=1;
    Method_Name = 'Current Classifier';
elseif nargin<4
    Method_Name = 'Current Classifier';
end

if Plot_Flag==1
    X_random = [0,1];
    Y_random = X_random;
    figure('units','inches','position',[0 5 5 5])
    box on
    hold on
    plot(ROC_X,ROC_Y,'LineStyle','-','LineWidth',2,'Color','r')
    plot(X_random,Y_random,'LineStyle',':','LineWidth',2,'Color','k')
    hold off
    axis equal tight
    legend({Method_Name,'Random Guessing'},'FontSize',12,'Location','southeast','FontWeight','bold')
    xlabel('False positive rate','FontSize',14)
    ylabel('True positive rate','FontSize',14)
    title(sprintf('ROC analysis\n AUC = %1.4f',AUC),'FontSize',14,'FontWeight','bold')
end