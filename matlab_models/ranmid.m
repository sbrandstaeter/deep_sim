%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  MATLAB code for the generation of self-affine randomly rough surfaces 
%  with the random midpoint displacement method. This method has been used 
%  for the analyses in:
%  
%  G. Zavarise, M. Borri-Brunetto, M. Paggi (2004) 
%  On the reliability of microscopical contact models, Wear, 257:229�245
%  http://dx.doi.org/10.1016/j.wear.2003.12.010
%
%  M. Paggi, M. Ciavarella (2010) The coefficient of proportionality k 
%  between real contact area and load, with new asperity models, Wear, 268:1020�1029
%  http://dx.doi.org/10.1016/j.wear.2009.12.038
%
%  M. Paggi, J.R. Barber (2011) Contact conductance of rough surfaces 
%  composed of modified RMD patches, Int. J. Heat Mass Transfer, 54:4664�4672
%  http://dx.doi.org/10.1016/j.ijheatmasstransfer.2011.06.011
%
%
%  Release date: 01/01/2004
%  Author: M. Paggi (C) 
%  Affiliation: IMT School for Advanced Studies Lucca
%               Piazza San Francesco 19, 55100 Lucca, Italy
%  Contact: marco.paggi@imtlucca.it
%
%  If you use this code, please refer to the articles above in your
%  publication.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function z=ranmid(n,H,k,r,rnd)

randn('state',rnd(k,r))
N=2^n;
x=1:1:2^n+1;

z(1,1)=0;
z(1,N+1)=0;
z(N+1,1)=0;
z(N+1,N+1)=0;

alpha=1/sqrt(0.09);

D=N;
d=N/2;

for i=1:n;

alpha=alpha/sqrt(2)^(H);

for j=d+1:D:N-d+1,
for k=d+1:D:N-d+1,
z(j,k)=(z(j+d,k+d)+z(j+d,k-d)+z(j-d,k+d)+z(j-d,k-d))/4+alpha*randn;
end;
end;

alpha=alpha/sqrt(2)^(H);

for j=d+1:D:N-d+1,
z(j,1)=(z(j+d,1)+z(j-d,1)+z(j,d+1))/3+alpha*randn;
z(j,N+1)=(z(j+d,N+1)+z(j-d,N+1)+z(j,N-d+1))/3+alpha*randn;
z(1,j)=(z(1,j+d)+z(1,j-d)+z(d+1,j))/3+alpha*randn;
z(N+1,j)=(z(N+1,j+d)+z(N+1,j-d)+z(N-d+1,j))/3+alpha*randn;
end;

for j=d+1:D:N-d+1,
for k=D+1:D:N-d+1,
z(j,k)=(z(j,k+d)+z(j,k-d)+z(j+d,k)+z(j-d,k))/4+alpha*randn;
end;
end;

for j=D+1:D:N-d+1,
for k=d+1:D:N-d+1,
z(j,k)=(z(j,k+d)+z(j,k-d)+z(j+d,k)+z(j-d,k))/4+alpha*randn;
end;
end;

D=D/2;
d=d/2;

end;
