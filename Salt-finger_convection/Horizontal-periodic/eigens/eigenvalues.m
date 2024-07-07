clear all;
close all;
clc;

% define parameters
kx = 0.2*pi; % wavenumber
ky = 0; % fixed =0 for 2D problem
Pr = 1; 
Ra = 1e6; 


N = 2; %?
[xi, DM] = chebdif(N+2, 4);
D2=DM(2:end-1,2:end-1,2)*2^2;
[~,dd4]=cheb4c(N+2);
D4=dd4*2^4;

I=eye(N,N);
O=zeros(N,N);


Laplacian=D2-(kx^2+ky^2)*I;
inv_Laplacian=inv(Laplacian);
Laplacian_square=D4-2*(kx^2+ky^2)*D2+(kx^2+ky^2)^2*I;
                    
A = [Pr*inv_Laplacian*Laplacian_square, inv_Laplacian*Pr*Ra*(-(kx^2+ky^2));
     I,                                 Laplacian];
                    
[eig_vec, eig_val] = eig(-A);
eig_val=diag(eig_val);

eig_val(find(real(eig_val)==Inf))=-Inf;


% plot eigenvalues
plot(real(eig_val),imag(eig_val),'o');
% xlim([-110,110]);
% ylim([-110,110]);
line([0 0], ylim,'Color','black');  %x-axis
line(xlim, [0 0],'Color','black');  %y-axis
xlabel('real') 
ylabel('imag') 