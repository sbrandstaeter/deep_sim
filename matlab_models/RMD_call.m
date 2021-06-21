clear;
clc;

%rng default
nn = 2;

H = 0.1;
rnd = 95.0129; 
%rnd = 46.1139;

z=ranmid(nn,H,1,1,rnd);

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

%% for
% % for i=1:5
% %     z_new=ranmid(nn,H,1,1,i);
% %     scalefactor = zref/(max(max(z_new))-mean(mean(z_new)));
% %     z_new = z_new*scalefactor;
% % 
% %     % setting minimum height to zero
% %     z_new = z_new-min(min(z_new));
% %     
% %     zmean = mean(z_new(:)); 
% %     rms = std(z_new(:));  
% %     zmax = max(max(z_new));
% %     sprintf("mean: %f ,std: %f ,max: %f",zmean, rms, zmax)
% % end