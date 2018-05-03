%Read ground-truth image
f=double(imread('office_3.jpg'));
f=f(1:256,1:256,:);
f=f/max(f(:));

stream = RandStream('mcg16807', 'Seed',0);
RandStream.setGlobalStream(stream);
stdn=.1;
noise=stdn*randn(size(f));
%Add noise
y=f+noise;


%Denoise color image
lambda=0.08; % regularization parameter
[xST,P,fun_val,ISNR]=proxSTV(y,lambda,'verbose',true,'img',f,'maxiter',50,'kernel',fspecial('gaussian',[3 3],0.5),'L',8/1.25,'snorm','nuclear','project',@(x)BoxProjection(x,[0 1]),'showfig',1);
