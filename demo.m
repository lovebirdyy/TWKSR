% a numerial example
% Copyright @ Yang Wang, 2021
clc
clear all
close all   

all=load('numbdata.mat')
X=all.data;
X=[X(1:100,1:3);X(5001:5100,1:3);X(10001:10100,1:3)]'; 
a=100; %sample number of each mode
C=twksr(X,2,20); 
%% spectral clustering on C
Cc = BuildAdjacency(thrC(C,1));
grps = SpectralClustering(Cc,3);
%% plot the mode identification result
figure(1)
scatter(1:3*a,grps)
set(gca,'ytick',1:3);
set(gca,'yticklabel',{'1','2','3'});  
xlabel('Samples'),ylabel('Mode number');
title('TWKSR'); 
