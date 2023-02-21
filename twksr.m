function C = twksr( Y,sigma,l )
%% This matlab code implements ADMM method for TWKSR algorithm
%-------------------------------------------------------------
% min |C|_1+  (lambda1)/2 * |phi(Y)-phi(Y)C|_F^2 +  lambda2 *|W * C|_1
% s.t., diag(C)=0, C^T * 1=1
%-------------------------------------------------------------
% inputs:
%        Y -- m*N train data matrix; N samples; m variables
%        sigma -- width of Gaussian kernel 
%        l -- length of window
% outputs:
%        C -- N*N representation matrix
%----------------------------------------------------------------------------
% created by Yang Wang, Huazhong University of Science and Technology, China
% 2021-5-16

%% If you use this code, please cite my work
%@ARTICLE{9511822,
%  author={Wang, Yang and Zheng, Ying and Wang, Zhaojing and Yang, Weidong},
%  journal={IEEE Transactions on Industrial Informatics}, 
%  title={Time-Weighted Kernel-Sparse-Representation-Based Real-Time Nonlinear Multimode Process Monitoring}, 
%  year={2022},
% volume={18},
%  number={4},
%  pages={2411-2421},
%  doi={10.1109/TII.2021.3104111}}

N = size(Y,2);  %sample number of training dataset 
%% parameter
thr = 2*10^-4;  
alpha = 20;  
maxIter = 200;  
thr1=thr;
thr2=thr;
thr3=thr;
thr4=thr;
if (length(alpha) == 1)
    alpha1 = alpha(1);
    alpha2 = alpha(1);
elseif (length(alpha) == 2)
    alpha1 = alpha(1);
    alpha2 = alpha(2);
end 

mu1 = 10*alpha1 * 1/computeLambda_mat(Y);  %lambda1  
mu3 = 5;   %lambda2    
mu2 = 20;  %rho  

KK = computeKM(Y',Y',sigma);  %kernel function 

tic;
 % initialization
    Acoe = inv(mu1*KK+2*mu2*ones(N)+mu2*eye(N));
    Cold = zeros(N);
    Aold = zeros(N);
    Bold = zeros(N);
    Lambda= zeros(N,1);  %delta (vector)
    Lambda1 = zeros(N);  %delta1 (matrix)
    Lambda2 = zeros(N);  %delta2 (matrix)
    err1 = 10*thr1; 
    err2 = 10*thr2;
    err3 = 10*thr3;
    err4 = 10*thr4;
    %% time weighted matrix W
    for i=1:N
        for j=1:N
            if abs(i-j)>=l
                W(i,j)=log(abs(i-j));  %log           
            else
                W(i,j)=0;
            end
        end
    end
    
    i = 1;      
    % ADMM iterations
    while ( (err1(i) > thr1 || err2(i) > thr2 || err3(i) > thr3 || err4(i) > thr4) && i < maxIter )
        % updating A       
        A = Acoe * (mu1*KK+mu2*ones(N,N)+mu2*Cold+mu2*Bold-ones(N,1)*Lambda'-Lambda1-Lambda2); 
        A = A - diag(diag(A));
        % updating C
        C = max(0,(abs(A+Lambda1/mu2)  - 1/mu2*ones(N,N))) .* sign(A+Lambda1/mu2);
        C = C - diag(diag(C));
        % updating B
        B = max(0,(abs(A+Lambda2/mu2)  - mu3/mu2*W)) .* sign(A+Lambda2/mu2);
        B = B - diag(diag(B));
        % updating Lagrange multipliers
        Lambda = Lambda + mu2*(A'*ones(N,1)-ones(N,1));
        Lambda1 = Lambda1 + mu2 * (A - C);
        Lambda2 = Lambda2 + mu2 * (A - B);
        % computing errors
        err1(i+1) = errorCoef(A,C);
        err2(i+1) = errorCoef(A,Aold);
        err3(i+1) = errorCoef(A,B);
        err4(i+1)= max(abs(A'*ones(N,1)-ones(N,1)));
        err5(i+1) = sum(sum(abs(Y-Y*C)));
        %
        Aold=A;
        Bold=B;
        Cold = C;
        i = i + 1;
    end
toc;    

    fprintf('err1: %2.4f, err2: %2.4f, err3: %2.4f, err4: %2.4f, err5: %2.4f, iter: %3.0f \n',...
        err1(end),err2(end),err3(end),err4(end),err5(end),i);
    
end



