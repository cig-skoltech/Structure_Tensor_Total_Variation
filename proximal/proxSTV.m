function [x,P,fun_val,ISNR]=proxSTV(y,lambda,varargin)

%Proximal map of the Structure Tensor Total Variation (STV) regularizer
%under convex constraints
%
%  argmin  0.5*||y-x||^2+ lambda*||Dx||_(1,p) + i_C(x),
%
% D is the patch-based Jacobian and D(x)D^T(x)=J(x) where J(x) is the
% Structure Tensor.
% i_C: indicator function of the closed convex set C.
%
% ========================== INPUT PARAMETERS (required) ==================
% Parameters    Values description
% =========================================================================
% y             Noisy image.
% lambda        Regularization penalty parameter.
% ======================== OPTIONAL INPUT PARAMETERS ======================
% Parameters    Values' description
%
% img           Original Image. (For the compution of the ISNR improvement)
% maxiter       Number of iterations (Default: 100)
% tol           Stopping threshold for denoising (Default:1e-4)
% optim         The type of gradient-based method used {'fgp'|'gp'}.
%               (Fast Gradient Projection or Gradient Projection)
%               (Default: 'fgp')
% L             Lipschitz constant of the dual objective. (Default: 8)
% P             Initialization of the dual variables (Default: zeros).
% verbose       If verbose is set to true then info for each iteration is
%               printed on screen. (Default: false)
% showfig       If showfig is set to true the result of the reconstruction 
%               in every iteration is shown on screen. (Default: false)
% project       A function handle for the projection onto the convex set C.
%               (Default: project=@(x)x, which means that there is no
%               constrain on the values of x.)
% bc            Boundary conditions for the differential operators.
%               {'symmetric'|'circular'|'zero'} (Default: 'symmetric')
% snorm         Specifies the type of the Hessian Schatten norm.
%               {'spectral'|'nuclear'|'frobenius'|'Sp'}. (Default:
%               'nuclear'). If snorm is set to Sp then the order of
%               the norm has also to be specified.
% order         The order of the Sp-norm. 1<order<inf. For order=1 set
%               snorm='nuclear', for order=2 set snorm='frobenius' and
%               for order=inf set snorm='spectral'.
% kernel        Smoothing kernel used for the construction of the Structure
%               Tensor. (Default: fspecial('gaussian',[3 3],0.5))
% =========================================================================
% ========================== OUTPUT PARAMETERS ============================
% x             Denoised image.
% P             The solution of the dual problem.
% fun_val       Evolution of the objective function during the iterations.
% ISNR          Evolution of the ISNR during the iterations.
% =========================================================================
%
% Author: stamatis@math.ucla.edu
%
% =========================================================================

[maxiter,L,tol,optim,verbose,img,project,P,bc,snorm,order,kernel,showfig]=...
  process_options(varargin,'maxiter',100,'L',8,'tol',1e-4,'optim','fgp',...
  'verbose',false,'img',[],'project',@(x)x,'P',[],...
  'bc','symmetric','snorm','nuclear','order',[],'kernel',...
  fspecial('gaussian',[3 3],0.5),'showfig',false);

if isequal(snorm,'Sp') && isempty(order)
  error('proxSTV: The order of the Sp-norm must be specified.');
end

if isequal(snorm,'Sp') && isinf(order)
  error('proxSTV: Try spectral norm for the Sp-norm instead.');
end

if isequal(snorm,'Sp') && order==1
  error('proxSTV: Try nuclear norm for the Sp-norm instead.');
end

if isequal(snorm,'Sp') && order < 1
  error('proxSTV: The order of the Sp-norm should be greater or equal to 1.');
end

[NGx,NGy] = size(kernel);

if ~all(mod([NGx,NGy], 2)) % if not all [NGx,NGy] are odd numbers
  error('proxSTV: The dimensions of the smoothing kernel must both be odd numbers');
end


if isempty(L)
  L=8/1.25;%Lipschitz constant
end

if isempty(P)
  [nx, ny, nc]=size(y);
  P=zeros([nx ny 2 NGx*NGy*nc]);
end

count=0;
w=sqrt(kernel);
ISNR=zeros(maxiter,1);
fun_val=ISNR;

if verbose
  fprintf('*********************************************************\n');
  fprintf('**           Denoising with STV Regularizer           **\n');
  fprintf('*********************************************************\n');
  fprintf('#iter     relative-dif   \t fun_val         Duality Gap        ISNR\n')
  fprintf('====================================================================\n');
end
switch optim
  case 'fgp'
    t=1;
    F=P;
    for i=1:maxiter
      K=y-lambda*AdjJacobianOp2D(F,w,bc);
      Pnew=F+(1/(L*lambda))*JacobianOp2D(project(K),w,bc);
      Pnew=projectLB(Pnew,snorm,order);
      
      re=norm(Pnew(:)-P(:))/norm(Pnew(:));%relative error
      if (re<tol)
        count=count+1;
      else
        count=0;
      end
      
      tnew=(1+sqrt(1+4*t^2))/2;
      F=Pnew+(t-1)/tnew*(Pnew-P);
      P=Pnew;
      t=tnew;
      
      if verbose
        if ~isempty(img)
          k=y-lambda*AdjJacobianOp2D(P,w,bc);
          x=project(k);
          fun_val(i)=cost(y,x,w,lambda,bc,snorm,order);
          dual_fun_val=dualcost(y,k,project);
          dual_gap=(fun_val(i)-dual_fun_val);
          ISNR(i)=20*log10(norm(y(:)-img(:))/norm(x(:)-img(:)));
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val(i),dual_gap,ISNR(i));
        else
          k=y-lambda*AdjJacobianOp2D(P,w,bc);
          x=project(k);
          fun_val(i)=cost(y,x,w,lambda,bc,snorm,order);
          dual_fun_val=dualcost(y,k,project);
          dual_gap=(fun_val(i)-dual_fun_val);
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f\n',i,re,fun_val(i),dual_gap);
        end
      end
      
      if showfig
        fh=figure(1);
        figure(fh);
        if verbose
          msg=['iteration: ' num2str(i) ' ,ISNR: ' num2str(ISNR(i))];
          set(fh,'name',msg);imshow(x,[]);
        else
          imshow(x,[]);
        end
      end
      
      if count >=5
        fun_val(i+1:end)=[];
        ISNR(i+1:end)=[];
        break;
      end    
    end
    
  case 'gp'
    
    for i=1:maxiter
      K=y-lambda*AdjJacobianOp2D(P,w,bc);
      Pnew=P+(1/(L*lambda))*JacobianOp2D(project(K),w,bc);
      Pnew=projectLB(Pnew,snorm,order);
      
      re=norm(Pnew(:)-P(:))/norm(Pnew(:));%relative error
      if (re<tol)
        count=count+1;
      else
        count=0;
      end
      
      P=Pnew;
      
      if verbose
        if ~isempty(img)
          k=y-lambda*AdjJacobianOp2D(P,w,bc);
          x=project(k);
          fun_val(i)=cost(y,x,w,lambda,bc,snorm,order);
          dual_fun_val=dualcost(y,k,project);
          dual_gap=(fun_val(i)-dual_fun_val);
          ISNR(i)=20*log10(norm(y(:)-img(:))/norm(x(:)-img(:)));
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f \t %2.8f\n',i,re,fun_val(i),dual_gap,ISNR(i));
        else
          k=y-lambda*AdjJacobianOp2D(P,w,bc);
          x=project(k);
          fun_val(i)=cost(y,x,w,lambda,bc,snorm,order);
          dual_fun_val=dualcost(y,k,project);
          dual_gap=(fun_val(i)-dual_fun_val);
          % printing the information of the current iteration
          fprintf('%3d \t %10.5f \t %10.5f \t %2.8f\n',i,re,fun_val(i),dual_gap);
        end
      end
      
      if showfig
        fh=figure(1);
        figure(fh);
        if verbose && ~isempty(img)
          fh=figure(1);
          figure(fh);
          msg=['iteration: ' num2str(i) ' ,ISNR: ' num2str(ISNR(i))];
          set(fh,'name',msg);imshow(x,[]);
        else
          imshow(x,[]);
        end
      end
      
      if count >=5
        fun_val(i+1:end)=[];
        ISNR(i+1:end)=[];
        break;
      end    
    end
end

x=project(y-lambda*AdjJacobianOp2D(P,w,bc));

function Ap=projectLB(A,snorm,order)

if nargin < 3
  order=[];
end

switch snorm
  case 'spectral'
    Ap=projectSpMat2xNc(A,1,1);   
    
  case 'frobenius'
    Ap=projectSpMat2xNc(A,2,1);
    
  case 'nuclear'
    Ap=projectSpMat2xNc(A,inf,1);
    
  case 'Sp'
    Ap=projectSpMat2xNc(A,order/(order-1),1);
    
  otherwise
    error('proxSTV::Unknown type of norm.');
end


function [Q,STnorm]=cost(y,f,w,lambda,bc,snorm,order)


if nargin < 7
  order=[];
end

D=StructureTensorOp2D(f,w.^2,bc);
D_trace=D(:,:,1)+D(:,:,3);%trace of D
D_det=sqrt((D(:,:,1)-D(:,:,3)).^2+4*D(:,:,2).^2);%sqrt of determinant of D

switch snorm
  case 'spectral'
    STnorm=max(sqrt((D_trace(:)+D_det(:))/2),sqrt(abs(D_trace(:)-D_det(:))/2));
    STnorm=sum(STnorm(:));
  case 'frobenius'
    STnorm=sqrt(D_trace(:));
    STnorm=sum(STnorm(:));
  case 'nuclear'
    STnorm=sqrt((D_trace(:)+D_det(:))/2)+sqrt(abs(D_trace(:)-D_det(:))/2);
    STnorm=sum(STnorm(:));
  case 'Sp'
    STnorm=(((D_trace(:)+D_det(:))/2).^(order/2)+(abs(D_trace(:)-D_det(:))/2).^(order/2)).^(1/order);
    STnorm=sum(STnorm(:));
  otherwise
    error('proxSTV::Unknown type of norm.');
end


Q=0.5*norm(y(:)-f(:),2)^2+lambda*STnorm;

function Q=dualcost(y,f,project)
r=f-project(f);
Q=0.5*(sum(r(:).^2)+sum(y(:).^2)-sum(f(:).^2));


