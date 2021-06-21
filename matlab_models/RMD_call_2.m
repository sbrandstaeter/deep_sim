clear;
clc;

%rng default
nn = 2;

H = 0.1;
rnd = 95.0129; 
%rnd = 46.1139;
%rnd = 1;

z=ranmid_2(nn,H,rnd);

zref = 50; % reference for the scaling, former value = 25
scalefactor = zref/(max(max(z))-mean(mean(z)));
z = z*scalefactor;


%% setting minimum height to zero
z = z-min(min(z));         

zmean = mean(z(:)); 
rms = std(z(:));  
zmax = max(max(z));

%% Save surface to file.dat
res=string(nn);
save(strcat('sup',res,'_2','.dat'),'z','-ascii');
