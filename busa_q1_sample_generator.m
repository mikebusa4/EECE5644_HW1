%Author: Michael Busa
%Homework #1: Q1 Sample Generator
%Date: 2/10/2020
%Purpose: Using given class priors and gaussian parameters, generate 10000
%         samples to use in the following three parts of the problem.


clear
N=10000;
cP0 = .8;
cP1 = .2;

m0 = [-.1;0];
m1 = [.1;0];

s0=[1 -.9;-.9 1];
s1=[1 .9;.9 1];

data_set0 = zeros(3,N);
data_set1 = zeros(3,N);
data_set = zeros(3,N);

c0_num = 0;
c1_num = 0;

for n=1:N
    k=rand();
    if k<cP0
        c0_num=c0_num+1;
        data_set0(1:2,c0_num) = randGaussian(1,m0,s0);
        data_set0(3,c0_num) = 0;
    else
        c1_num=c1_num+1;
        data_set1(1:2,c1_num) = randGaussian(1,m1,s1);
        data_set1(3,c1_num) = 1;
    end
end

data_set(:,1:c0_num) = data_set0(:,1:c0_num);
data_set(:,c0_num+1:N) = data_set1(:,1:c1_num);
        

figure(1),
scatter(data_set0(1,:),data_set0(2,:),'or')
hold on
scatter(data_set1(1,:),data_set1(2,:),'b+')
c0_num
c1_num
hold off