function D = StructureTensorOp2D(u,G,bc)
% Operator of the tensor Total Variation (and related functionals) 
%
% u: Nx x Ny x Nc array of a vector-valued image with Nc-channels, defined
% on a Nx x Ny pixel grid
%
% G: NGx x NGy array of the convolution kernel G (all elements of G must 
% be >=0). NGx , NGy must be odd numbers, since the origin is considered 
% to correspond to the middle element of G. 
%
% bc: boundary condition type: 'symmetric' |'circular'|'zero'.
%
% (OUTPUT):
% D: 3D array with dimensions (Nx,Ny,3), which for each pixel (i,j)
% the last dimension of D contains the 3 unique elements of the Structure 
% tensor, i.e, D(i,j,:)=[G*fx(i,j)^2,G*fx(i,j)*fy(i,j),G*fy(i,j)^2] 
% where 
%   fx(i,j)^2=fx(i,j,1)^2+...+fx(i,j,Nc)^2
%   fx(i,j)*fy(i,j)=fx(i,j,1)*fy(i,j,1)+...+fx(i,j,Nc)*fy(i,j,Nc)
%   fy(i,j)^2=fy(i,j,1)^2+...+fy(i,j,Nc)^2

% Author: stamatis@math.ucla.edu

if nargin < 3
  bc='symmetric';
end


if isequal(bc,'zero')
  bc=0;
end

[Nx,Ny,~] = size(u);

[NGx,NGy] = size(G);

if ~all(mod([NGx,NGy], 2)) % if not all [NGx,NGy] are odd numbers
    error('The dimensions of the kernel G must both be odd numbers');
end

grad_u  = gradOp2D(u,bc);

D = zeros(Nx,Ny,3);
D(:,:,1)=imfilter(squeeze(sum(grad_u(:,:,:,1).^2,3)),G,'conv',bc);
D(:,:,2)=imfilter(squeeze(sum(grad_u(:,:,:,1).*grad_u(:,:,:,2),3)),G,'conv',bc);
D(:,:,3)=imfilter(squeeze(sum(grad_u(:,:,:,2).^2,3)),G,'conv',bc);


function Df=gradOp2D(f,bc)

[r,c,k]=size(f);
Df=zeros(r,c,k,2);
Df(:,:,:,1)=shift(f,[-1,0,0],bc)-f; %f(i+1,j,l)-f(i,j,l)
Df(:,:,:,2)=shift(f,[0,-1,0],bc)-f; %f(i,j+1,l)-f(i,j,l)
