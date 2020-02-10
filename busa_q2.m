%Author: Michael Busa
%Homework #1: Q2: Minimum Classification rule
%Date: 2/10/2020
%Purpose: Generate a dataset based on given priors and gaussian parameters
%         Then use the dataset to evaulate the pdfs, and find a
%         classification rule. Plot true data set and dataset with
%         classification rule.


clear
%Assign Priors
classPrior0 = .35;
classPrior1 = .65;

classPrior0a = .7;
classPrior0b = .3;
classPrior1a = .4;
classPrior1b = .6;

%Number of samples in each class
classLabel0=0;
classLabel1=0;

%Assign MUs and SIGMAS
class0a_MU = [8;6];
class0a_SIGMA = [7 2;2 5]/3;

class0b_MU = [2;5];
class0b_SIGMA = [8  -1;-1 5]/8;

class1a_MU = [1;-1];
class1a_SIGMA = [4  1;1 12]/7;

class1b_MU = [4;2];
class1b_SIGMA = [13 2;2 6]/11;

%Determine random number of samples in each class
for n=1:1000
   k = rand();
   if k<classPrior0
       classLabel0=classLabel0+1;
   else
       classLabel1=classLabel1+1;
   end
end

label0 = zeros(classLabel0,2);
label1 = zeros(classLabel1,2);
samples = zeros(1000,3);

%Generate samples for class 0
for n=1:classLabel0
   k = rand();
   if k<classPrior0a
      label0(n,:) = randGaussian(1,class0a_MU,class0a_SIGMA);
      samples(n,1:2) = label0(n,:);
      samples(n,3) = 0;
   else
      label0(n,:) = randGaussian(1,class0b_MU,class0b_SIGMA);
      samples(n,1:2) = label0(n,:);
      samples(n,3) = 0;
   end
end

%Generate samples for class 1
for n=1:classLabel1
   k = rand();
   if k<classPrior1a
      label1(n,:) = randGaussian(1,class1a_MU,class1a_SIGMA);
      samples(n+classLabel0,1:2) = label1(n,:);
      samples(n+classLabel0,3) =1;
   else
      label1(n,:) = randGaussian(1,class1b_MU,class1b_SIGMA);
      samples(n+classLabel0,1:2) = label1(n,:);
      samples(n+classLabel0,3) =1;
   end
end

samples = samples'; %transpose to work with future functions
x = samples(1:2,:); %x now holds all samples
labels = samples(3,:); %labels holds the labels for eacah corresponding sample

%Plot Samples
figure(2),
scatter(label0(:,1),label0(:,2),'or')
hold on
scatter(label1(:,1),label1(:,2),'+b')
axis equal
legend('Class 0', 'Class 1', 'Location', 'Southeast')
title('Original Data, True Labels')
xlabel('x1')
ylabel('x2')
hold off

%calculate threshold
lam=[0 1;1 0]; %loss values 0-1 loss
gam=classPrior0/classPrior1; %with 0-1 loss, gamma is just division of the class priors ((1-0/1-0)*(p0/p1))

%Calculate class pdfs using evalGaussian function
pdf0 = classPrior0a*evalGaussian(x,class0a_MU,class0a_SIGMA)+classPrior0b*evalGaussian(x,class0b_MU,class0b_SIGMA);
pdf1 = classPrior1a*evalGaussian(x,class1a_MU,class1a_SIGMA)+classPrior1b*evalGaussian(x,class1b_MU,class1b_SIGMA);
discriminant = log(pdf1)-log(pdf0);

%Decide on a class
dec = (discriminant>=log(gam));
true_neg = find(dec==0 & labels==0); 
false_pos = find(dec==1 & labels==0); 
false_neg = find(dec==0 & labels==1); 
true_pos = find(dec==1 & labels==1); 

p00 = length(true_neg)/classLabel0; % probability of true negative
p10 = length(false_pos)/classLabel0; % probability of false positive
p01 = length(false_neg)/classLabel1; % probability of false negative
p11 = length(true_pos)/classLabel1; % probability of true positive

figure(1)
hold on
plot(x(1,true_neg),x(2,true_neg),'oc')
plot(x(1,false_pos),x(2,false_pos),'*m')
plot(x(1,false_neg),x(2,false_neg),'xm')
plot(x(1,true_pos),x(2,true_pos),'+c')
axis equal,

% Prepare figure for boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
GridValues = log(evalGaussian([h(:)';v(:)'],class1a_MU,class1a_SIGMA)+ evalGaussian([h(:)';v(:)'],class1b_MU,class1b_SIGMA))-log(evalGaussian([h(:)';v(:)'],class0a_MU,class0a_SIGMA)+evalGaussian([h(:)';v(:)'],class0b_MU,class0b_SIGMA)) - log(gam);
min_vals = min(GridValues);max_vals = max(GridValues);
discriminantGrid = reshape(GridValues,91,101);

%Plot the boundary
contour(horizontalGrid,verticalGrid,discriminantGrid,[min_vals*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*max_vals]); % plot equilevel contours of the discriminant function 
legend('Class 0 Correct','Class 0 Incorrect','Class 1 Incorrect','Class 1 Correct', 'Decision Boundary', 'Location', 'Southeast')
title('Original Data, Classifier Decisions'),
xlabel('x1'), ylabel('x2')




