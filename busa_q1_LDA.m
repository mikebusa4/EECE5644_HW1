%Author: Michael Busa
%Homework #1: Q1 part 3: Fisher LDA
%Date: 2/10/2020
%Purpose: Import Data from the saved .mat file, evaulate the gaussian pdfs 
%         with the Fisher LDA Model and find the ROC curve for true 
%         and false positive probs


%Load in saved dataset
clear
clc
N=10000;
load('q1_10000_samples.mat');
%Separate actual data from true class labels for each point
samples = data_set(1:2,:);
true_class_labels = data_set(3,:);

%Prepare to separate data
class0_data = zeros(2,N);
class1_data = zeros(2,N);
class0_sample_num = 0;
class1_sample_num = 0;

%Class priors, gaussian parameters
class0_prior = .8;
class1_prior = .2;

%Separate data by classes, count how many data points per class
for n=1:N
    if(true_class_labels(n))
        class1_data(:,n)=samples(:,n);
        class1_sample_num=class1_sample_num+1;
    else
        class0_data(:,n)=samples(:,n);
        class0_sample_num=class0_sample_num+1;
    end
end

%Calculate MUs and SIGMAs based on the samples
sample_mu0 = sum(class0_data,2)./class0_sample_num;
sample_mu1 = sum(class1_data,2)./class1_sample_num;

sample_sig0 = cov(class0_data(1,:),class0_data(2,:))./class0_prior;
sample_sig1 = cov(class1_data(1,:),class1_data(2,:))./class1_prior;

%Find Projection vector w
Sb = (sample_mu0-sample_mu1)*(sample_mu0-sample_mu1)';
Sw = sample_sig0 + sample_sig1;
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

%Projected data
y0 = w'*class0_data(:,1:class0_sample_num);
y1 = w'*class1_data(:,class0_sample_num+1:N);
y = [y0,y1];

%Plot projection
figure(3)
scatter(y1,zeros(1,class1_sample_num),'bo')
hold on
scatter(y0,zeros(1,class0_sample_num),'ro')
hold off
title('Projection of dataset on W')
xlabel('x1'),ylabel('x2')
legend('Class 1', 'Class 0', 'Location', 'Northeast')

%Class Decision
increments = 10000;
threshold_values = zeros(increments,1);
row_counter=1;
min_pError = 1;
min_probs = [0,0];

%-inf to 0
for i=10000:-1:5000
    lambda = [1-(i/increments) (i/increments); (i/increments) 1]; %By changing the lambdas, the threshold gradually increases
    gamma = ((lambda(1,1)-lambda(2,1))/(lambda(2,2)-lambda(1,2)))*(class0_prior/class1_prior);
    threshold_values(row_counter) = gamma;
    decision = (y>=gamma);
    false_pos = find(decision==1 & true_class_labels==0); p10 = length(false_pos)/class0_sample_num; % probability of false positive
    true_pos = find(decision==1 & true_class_labels==1); p11 = length(true_pos)/class1_sample_num; % probability of true positive
    probabilities(row_counter,:) = [p10,p11];
    row_counter = row_counter+1;
    pError = p10 + (1-p11);
    if pError<min_pError
        min_pError = pError;
        min_probs = [p10,p11];
    end
end

%0-inf
for i=10000:-1:1
    lambda = [1, 1-(i/increments); (i/increments) 1]; %By changing the lambdas, the threshold gradually increases
    gamma = ((lambda(1,1)-lambda(2,1))/(lambda(2,2)-lambda(1,2)))*(class0_prior/class1_prior);
    threshold_values(row_counter) = gamma;
    decision = (y>=gamma);
    false_pos = find(decision==1 & true_class_labels==0); p10 = length(false_pos)/class0_sample_num; % probability of false positive
    true_pos = find(decision==1 & true_class_labels==1); p11 = length(true_pos)/class1_sample_num; % probability of true positive
    probabilities(row_counter,:) = [p10,p11];
    row_counter = row_counter+1;
    pError = p10 + (1-p11);
    if pError<min_pError
        min_pError = pError;
        min_probs = [p10,p11];
    end
end

min_pError = (min_probs(1)+(1-min_probs(2)))/2;


%Plot the ROC Curve with the minimum p(error) point marked
figure(1)
scatter(probabilities(:,1),probabilities(:,2),'b.') %plot of false positive vs true positive
hold on
plot(min_probs(1),min_probs(2), '-p','Markersize', 15, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red')
txt = strcat('  \leftarrow Min P(error) at [',num2str(min_probs(1)));
txt = strcat(txt,',');
txt = strcat(txt,num2str(min_probs(2)));
txt = strcat(txt,']');
text(min_probs(1),min_probs(2),txt);
title('ROC Curve of Classifier')
xlabel('p(False Positive)'),ylabel('p(True Positive)')
legend('ROC', 'Min P(error)', 'Location', 'Northeast')
axis equal

%Plot of the data set
figure(2),
scatter(class0_data(1,:),class0_data(2,:),'or')
hold on
scatter(class1_data(1,:),class1_data(2,:),'b+')
title('Saved Dataset from Given Gaussians')
legend('Class 0', 'Class 1', 'Location', 'North')
xlabel('x1'),ylabel('x2')
hold off

fprintf('The estimate of the minimum p(Error) is %.4f',min_pError);