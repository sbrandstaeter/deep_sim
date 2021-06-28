%programma analisi statistica superfici scabre

clc
clf
clear all
format long

nsupt=1;     %number of surfaces to be analysed at the same time
nnodi=5;   %number of heights per side

nprof=nnodi;   %number of profiles


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Reading surface data x, y, z
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for g=1:nsupt
  xxx=sprintf('sup2_dat.dat');	
	fp=fopen(['',xxx],'r');
	v=(fscanf(fp,'%f',[3,inf]));
	v=v';

delta=v(2,1); % sampling interval

lun=length(v(:,1));
for k=1:lun
	i=ceil(k/nnodi); %row index
	j=k-nnodi*(i-1); %column index
	x(i,j)=delta*i;
	y(i,j)=delta*j;
	z(i,j)=v(k,3);
end

%z=z-min(min(z));

% figure(5*(g-1)+g)   % surface picture
% axes('FontSize',24);
% view([-37.5 30]);
% grid('on');
% hold('all');
% mesh(x,y,z)
% xlabel('x (\mum)','FontSize',30);
% ylabel('y (\mum)','FontSize',30);
% zlabel('z (\mum)','FontSize',30);

m0(g)=std2(z);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Compute profile statistics: slopes, maxima 2D (peaks) curvatures
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% evaluate rms slopes in x and y directions

slopey=[];
ny=0;
for j=2:nnodi-1
	 for i=2:nprof-1
            ny=ny+1;
    	    slopey(i-1,j-1)=(z(i+1,j)-z(i-1,j))/(2*delta);
            slopeyv(ny)=slopey(i-1,j-1);
	 end
end

slopex=[];
nx=0;
for i=2:nprof-1
	for j=2:nnodi-1
            nx=nx+1;
  	    slopex(i-1,j-1)=(z(i,j+1)-z(i,j-1))/(2*delta);
            slopexv(nx)=slopex(i-1,j-1); 
	end
end

% evaluates 2D maxima (peaks)

curvp=[];
zpeak=[];
n_peaks=0;
for j=2:nnodi-1
	for i=2:nprof-1
    	if (z(i,j)>z(i,j-1) & z(i,j)>z(i,j+1));
      	   n_peaks=n_peaks+1;
       	   curvp(n_peaks)=-(z(i,j+1)-2*z(i,j)+z(i,j-1))/(delta^2);
           zpeak(n_peaks)=z(i,j);
    	end
    end	
end

% statistics profile slopes
rms_slopex(g)=std(slopexv);
rms_slopey(g)=std(slopeyv);

m2x(g)=rms_slopex(g)^2;
m2y(g)=rms_slopey(g)^2;


% statistics heights of peaks
mean_z_peaks(g)=mean(zpeak);
rms_z_peaks(g)=std(zpeak);
ks_z_peaks(g)=kurtosis(zpeak);
sk_z_peaks(g)=skewness(zpeak);

% statistics curvatures of peaks
mean_curv_peaks(g)=mean(curvp);
rms_curv_peaks(g)=std(curvp);
ks_curv_peaks(g)=kurtosis(curvp);
sk_curv_peaks(g)=skewness(curvp);

m4(g)=rms_curv_peaks(g)^2;

density_peaks(g)=n_peaks/(nnodi*nprof); %density of peaks

alfa_x(g)=m0(g)*m4(g)/m2x(g)^2;
alfa_y(g)=m0(g)*m4(g)/m2y(g)^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Compute asperity (3D maxima) heights, curvatures and statistics
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Curvx=[];
Curvy=[];
R=[];

for i=2:nprof-1
   for j=2:nnodi-1
       if (z(i,j)>z(i,j-1) & z(i,j)>z(i,j+1))
       Curvy(i,j)=-2*(-delta*z(i,j-1)+2*delta*z(i,j)-delta*z(i,j+1))/(-delta*y(i,j-1)^2+2*delta*y(i,j)^2-delta*y(i,j+1)^2);
       else
       Curvy(i,j)=0;
       end
   end
end

for j=2:nnodi-1
   for i=2:nprof-1
       if (z(i,j)>z(i-1,j) & z(i,j)>z(i+1,j))
       Curvx(i,j)=-2*(-delta*z(i-1,j)+2*delta*z(i,j)-delta*z(i+1,j))/(-delta*x(i-1,j)^2+2*delta*x(i,j)^2-delta*x(i+1,j)^2);
       else
       Curvx(i,j)=0;
       end
   end
end

ns=0;

H=[];
ro=[];
curv=[];

for i=2:nprof-1
   for j=2:nnodi-1
       if (Curvx(i,j)*Curvy(i,j)~=0)
       R(i,j)=1/sqrt(Curvx(i,j)*Curvy(i,j));
       ns=ns+1;
       H(ns)=z(i,j);
       ro(ns)=R(i,j);
       curv(ns)=1/R(i,j);
       else
       R(i,j)=0;
       end
   end
end

% statistics of asperity (3D maxima) heights

mean_z_asperities(g)=mean(H);
rms_z_asperities(g)=std(H);
ks_z_asperities(g)=kurtosis(H);
sk_z_asperities(g)=skewness(H);

% statistics of asperity (3D maxima) curvatures

mean_curv_asperities(g)=mean(curv(1:ns));
rms_curv_asperities(g)=std(curv(1:ns));
ks_curv_asperities(g)=kurtosis(curv(1:ns));
sk_curv_asperities(g)=skewness(curv(1:ns));

density_asperities(g)=ns/(nnodi*nprof);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        Histograms of slopes
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(5*(g-1)+2)
histfit(slopexv,100);

figure(5*(g-1)+3)
histfit(slopeyv,100);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Histograms of curvatures & heights
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(5*(g-1)+4)
[n1,ctr1] = hist((H-mean_z_asperities(g))/rms_z_asperities(g),100);
[n2,ctr2] = hist(curv/rms_curv_asperities(g),100);
subplot(2,2,2);
%contour((H-mean_z_asperities(g))/rms_z_asperities(g),curv/rms_curv_asperities(g));
plot((H-mean_z_asperities(g))/rms_z_asperities(g),curv/rms_curv_asperities(g),'.k');
hold on
%axis([0 12 -8 8]);
h1 = gca;
%title('');
xlabel('Normalized asperity height','fontsize',20);
ylabel('Dimensionless asperity curvature','fontsize',20);
subplot(2,2,4);
bar(ctr1,n1,1);
%axis([0 12 -max(n1)*1.1 0]);
%axis('off');
h2 = gca;
subplot(2,2,1);
barh(ctr2,n2,1);
%axis([-max(n2)*1.1 0 -8 8]);
%axis('off');
h3 = gca;
set(h1,'Position',[0.35 0.35 0.55 0.55]);
set(h2,'Position',[.35 .1 .55 .15]);
set(h3,'Position',[.1 .35 .15 .55]);
colormap([.8 .8 1]);
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    Power Spectral Density function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Sx=[];
Sy=[];
ascx=[];
ascy=[];
qx=[];
qy=[];
risx=[];
risy=[];


w=fft2(z);

for i=1:nprof
    for j=1:nnodi
        rad2(i,j)=(real(w(i,j)))^2+(imag(w(i,j)))^2;
    end
end

Sy(1:nprof,1)=0;
for i=1:nnodi
    Sy(1:nprof,1)=Sy(1:nprof,1)+rad2(1:nprof,i);
end

Sy=Sy/nnodi;

Sx(1,1:nnodi)=0;
for i=1:nprof
    Sx(1,1:nnodi)=Sx(1,1:nnodi)+rad2(i,1:nnodi);
end

Sx=Sx'/nprof;

ascy(1:nprof/2)=2:2:nprof;
ascx(1:nnodi/2)=2:2:nnodi;

for i=1:length(ascy)
    qy(i)=2*pi*ascy(i)/(nprof*delta);
end

for i=1:length(ascx)
    qx(i)=2*pi*ascx(i)/(nnodi*delta);
end


risx=[qx',Sx(1:nprof/2)];
risy=[qy',Sy(1:nprof/2)];
save Sx.dat  risx -ASCII -DOUBLE -TABS
save Sy.dat  risy -ASCII -DOUBLE -TABS
	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    Creating vectors 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     omega_x=risx(2:length(risx),1);
     PSD_x=risx(2:length(risx),2);  
     omega_y=risy(2:length(risy),1);
     PSD_y=risy(2:length(risy),2);  

     log_omega_x=log10(omega_x);
     log_PSD_x=log10(PSD_x); 

     log_omega_y=log10(omega_y);
     log_PSD_y=log10(PSD_y);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Fitting and plotting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
    
      [fitx,gof]=fit(log_omega_x,log_PSD_x,'a*x+b');
      R(1)=gof.rsquare;
      coefx=coeffvalues(fitx);
      a1=coefx(1);
      b1=coefx(2);
      [fity,gof]=fit(log_omega_y,log_PSD_y,'c*x+d') ;
      coefy=coeffvalues(fity);
      a2=coefy(1);
      b2=coefy(2);
      R(2)=gof.rsquare;
      figure(5*(g-1)+5)
      plot (log_omega_x,log_PSD_x,'r:')
      hold on      
      plot (fitx,'r-')
      plot(log_omega_y,log_PSD_y,'k:')
      plot(fity, 'k-')
      legend ('PSD direction x','Best-fit','PSD direction y','Best-fit');
      xlabel('log_{10}\omega (1/\mum)','fontsize',24);
      ylabel('log_{10} PSD (\mum^3)','fontsize',24);
      hold off




            
      Dx(g)=(5+a1)/2+1;
      Dy(g)=(5+a2)/2+1;
      Gx(g)=10^(b1);
      Gy(g)=10^(b2);

if max(R)==R(1)
D=Dx;
G=Gx;
%sprintf('Bestfit PSDx')
else
D=Dy;
G=Gy;
%sprintf('Bestfit PSDy')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              PSD correlations
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      

    for j=1:nnodi/2-1
        PSDx(j+(g-1)*(nnodi/2))= log_PSD_x (j);
        omegax(j+(g-1)*(nnodi/2))=log_omega_x(j);
        PSDy(j+(g-1)*(nnodi/2))= log_PSD_y (j);
        omegay(j+(g-1)*(nnodi/2))=log_omega_y(j);
        PSDm(j+(g-1)*(nnodi/2))=(PSDx(j+(g-1)*(nnodi/2))+PSDy(j+(g-1)*(nnodi/2)))/2;
    end


s=cell(1);
s={'r.','k.','b.'} ;
a=(g-1)*(nnodi/2)+1;
b=((nnodi/2)*g)-1;
figure(49) 
plot (omegax(a:b),PSDx(a:b), char(s(g)))
      xlabel('log_{10}\omega (1/\mum)','fontsize',24);
      ylabel('log_{10} PSD (\mum^3)','fontsize',24);
      
hold on
if nsupt==3
legend ('PSDx 10x','PSDx 20x','PSDx 100x'); 
elseif nsupt==2
legend ('PSDx 10x','PSDx 20x'); 
else
legend ('PSDx 10x'); 
end
figure(50)
plot (omegay(a:b),PSDy(a:b), char(s(g)))
      xlabel('log_{10}\omega (1/\mum)','fontsize',24);
      ylabel('log_{10} PSD (\mum^3)','fontsize',24);
hold on
if nsupt==3
legend ('PSDy 10x','PSDy 20x','PSDy 100x'); 
elseif nsupt==2
legend ('PSDy 10x','PSDy 20x'); 
else
legend ('PSDy 10x');

end
%figure(51)
%plot (omegay(a:b),PSDy(a:b), char(s(g)))
%      xlabel('log_{10}\omega (1/\mum)','fontsize',24);
%      ylabel('log_{10} PSD (\mum^3)','fontsize',24);
%hold on
%if nsupt==3
%legend ('PSD 10x','PSD 20x','PSD 100x'); 
%elseif nsupt==2
%legend ('PSD 10x','PSD 20x'); 
%else
%legend ('PSD 10x');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    Save statistical and fractal parameters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ris_PSD=[Dx',Gx',Dy',Gy',D',G'];
save Result.dat  ris_PSD  -ASCII -DOUBLE -TABS
save Sx.dat omegax PSDx -ASCII -DOUBLE -TABS
save Sy.dat omegay PSDy -ASCII -DOUBLE -TABS
 
ris_stat=[m0',rms_slopex',rms_slopey',mean_z_peaks',rms_z_peaks',ks_z_peaks',sk_z_peaks',mean_curv_peaks',rms_curv_peaks',ks_curv_peaks',sk_curv_peaks',density_peaks',mean_z_asperities',rms_z_asperities',ks_z_asperities',sk_z_asperities',mean_curv_asperities',rms_curv_asperities',ks_curv_asperities',sk_curv_asperities',density_asperities',alfa_x',alfa_y'];

save ris_stat.dat ris_stat -ASCII -DOUBLE -TABS
