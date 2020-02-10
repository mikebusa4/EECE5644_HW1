%Author: Michael Busa
%Homework #1: Q1 part 1
%Date: 2/10/2020
%Purpose: Import Data from the saved .mat file, evaulate the gaussian pdfs 
%         and find the ROC curve for true and false positive probs


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

mu0 = [-.1;0];
mu1 = [.1;0];

sig0=[1 -.9;-.9 1];
sig1=[1 .9;.9 1];

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

%Calculate class pdfs using evalGaussian function
pdf0 = evalGaussian(samples,mu0,sig0);
pdf1 = evalGaussian(samples,mu1,sig1);
discriminant = log(pdf1)-log(pdf0);

%Decision rule with a varying Threshold
increments = 10000;
probabilities = zeros(increments,2);
threshold_values = zeros(increments,1);
row_counter=1;
min_pError = 1;
for i=increments:-1:1
    lambda = [1 1-(i/increments); (i/increments) 1]; %By changing the lambdas, the threshold gradually increases
    gamma = ((lambda(1,1)-lambda(2,1))/(lambda(2,2)-lambda(1,2)))*(class0_prior/class1_prior);
    threshold_values(row_counter) = gamma;
    decision = (discriminant>=log(gamma));
    false_pos = find(decision==1 & true_class_labels==0); p10 = length(false_pos)/class0_sample_num; % probability of false positive
    true_pos = find(decision==1 & true_class_labels==1); p11 = length(true_pos)/class1_sample_num; % probability of true positive
    probabilities(row_counter,:) = [p10,p11];
    row_counter = row_counter+1;
    pError = p10 + (1-p11); %(1-p(true pos) + p(false pos))
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
%Print estimate of min pError
fprintf('The estimate of the minimum p(Error) is %.4f',min_pError);