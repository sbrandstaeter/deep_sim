clear variables
close all
clc

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

format longE

b = 1; g0 = 0.01;
mx = 2048;

dx = 2*b/mx;

x = (-b:dx:+b)'; %FEM mesh!

%% Random midpoint displacement
H = 0.8;
rnd = 5.468815192049838e-01;
Z = ranmid(log2(mx),H,1,1,rnd);
Z = (Z-min(min(Z)))/(max(max(Z))-min(min(Z)));
[ind1,ind2]=find(Z==0);
zx = Z(ind1,:)';
zy = Z(:,ind2);
z = g0*zx/max(zx);

%% Weierstrass profile
% g0 = 0.01*b;
% H = 0.8;
% D = 2-H;
% nwv = [0 1 2 3]; nw = length(nwv); 
% l0 = 2*b;
% gm = 3;
% phi = 0.5;
% % z = WM1d(x,nw,D,gm,l0,g0,phi);
% 
% for i=1:nw
%     z0(:,i) = WM1d(x,nwv(i),D,gm,l0,g0,phi);
% end
% z = sum(z0');

%% Analytic (test)
%     z = g0*x.^2;

%% Export to .dat file, red by FEAP
data= [x z];
cd ..
  save('prof_2048.dat','data','-ascii');
cd pre_proc

%% Plots
normal = [0 0 2.2083 1.6562]; % 8/6
big = [0 0 4.4167 4.4167];
full = [0 0 6.6389 4.4167];
landscape = [0 0 4.4167 1.6562];

size_fig = landscape;

error('### End of file ###')

figure(1); fig=gcf; ax=gca;
fig.Units='inches';
fig.Position=size_fig;
fig.Renderer='painters';                 %vectorial image, unclear
fig.PaperPosition=size_fig;
fig.PaperSize=size_fig(3:4);
fig.PaperOrientation='portrait';
plot(x/b,z/g0)
xlabel('$x/b$')
ylabel('$z/g_0$')
% legend({'$n_\mathrm{w}=1$','$n_\mathrm{w}=2$','$n_\mathrm{w}=3$','$n_\mathrm{w}=4$'},'Location','northwest','NumColumns',2)
grid on
colormap jet
ax.FontSize=9;
saveas(gcf,'fig3_RM','pdf')

