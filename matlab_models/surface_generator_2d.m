clear variables
close all
clc

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

format longE

bx = 1000; by = 1000;
mx = 125; my = mx;

dx = 2*bx/mx; dy = 2*by/my; 

xv = (-bx:dx:+bx)'; %FEM mesh!
yv = (-by:dy:+by)';

[X,Y] = meshgrid(xv,flip(yv));

Z = zeros(my+1,mx+1);

%% Paraboloid:
% R = 100;
% Z = (X.^2+(Y/2).^2)/(2*R);

%% Cone
% Z = sqrt((X.^2+Y.^2))/(2*R);

%% Flat punch 
% Z(1,:) = ones(1,mx+1);
% Z(end,:) = ones(1,mx+1);
% Z(:,1) = ones(my+1,1);
% Z(:,end) = ones(my+1,1);

%% Eggbox:
% A = dx;
% Z = A*(1-cos(pi*(X-bx/2)).^2.*cos(pi*(Y-by/2)).^2)/10;
% Z = Z-min(min(Z));

%% Wavy sphere
% A = dx;
% Z = A*(1-cos(4*bx*pi*X).*cos(4*by*pi*Y))/20+A*(X.^2+Y.^2);
% Z = Z-min(min(Z));

%% Weierstrass-Mandelbrot
g0 = 0.001;
G = 3;
lam0 = 1;
D = 2.25;
gam = 1.30;
nw = 8;
M = 10;
% rand('state',0);
% phi=rand(M,nw)*2*pi;
% 
phi =[5.969838364357776e+00     3.866875487145548e+00     3.637417956339715e-01     9.596891389177384e-02     5.266053499629358e+00     1.215363799876004e+00     3.119931056218566e+00     4.568587081268629e+00
      1.452286112413295e+00     4.975887157772878e+00     2.217135863717760e+00     4.692192790601789e+00     1.233987049555711e-01     4.286534934686075e+00     5.653416461278788e+00     1.943327387653772e+00
      3.812904404680654e+00     5.791921713751309e+00     5.109275788149660e+00     2.796623363629681e+00     4.280590649884930e+00     1.902324834496621e+00     5.162448270682600e+00     5.268426029683141e+00
      3.053517906941135e+00     4.638292920531081e+00     6.196037942239526e-02     5.854763668406071e+00     2.384349556827069e+00     3.403437200095806e+00     4.052091450414305e+00     3.569304540417405e+00
      5.600196568411134e+00     1.107512849241778e+00     8.726771488131218e-01     2.927928800843848e+00     5.226328516415172e+00     9.479628669946573e-01     5.139484360011052e+00     2.327377016611048e+00
      4.788395623925821e+00     2.549127316943229e+00     1.274011442064966e+00     2.630452184484021e+00     3.159266524985838e+00     4.385025487124833e+00     4.148332082028957e+00     4.415445097640593e+00
      2.868070926988294e+00     5.877729468744615e+00     1.248605533707795e+00     5.316965979094473e+00     4.457740230498158e+00     2.377387677454669e+00     2.148664764202539e+00     3.434207830500863e+00
      1.162618193865363e-01     5.761080504951651e+00     3.793740033856142e+00     3.299630448813341e+00     2.694810208272050e+00     5.403612279828801e+00     1.820401491953358e+00     2.795264765455889e+00
      5.161053425911984e+00     2.577803736539035e+00     1.710207170962954e+00     1.273270900127680e+00     1.913967363825512e+00     5.363673374378772e+00     2.143782422251754e+00     4.364094679890071e+00
      2.794153644957320e+00     5.614965602403844e+00     1.249186886053974e+00     4.223164266322532e+00     1.191629640039956e+00     3.729465770956049e+00     3.355717436424364e+00     3.903806685015569e+00];

Z = WM2d(X,Y,G,lam0,D,gam,nw,M,phi);
Z = g0*(Z-min(min(Z)))/(max(max(Z))-min(min(Z))); %(scaling between 0 and g0)

%% Weierstrass-Mandelbrot paraboloid
% A = dx;
% G = 2;
% lam0 = bx/4;
% D = 2.8;
% gam = 10;
% nw = 4;
% M = 4;
% 
% Z = A/10*WM2d(X,Y,G,lam0,D,gam,nw,M)+A*(X.^2+Y.^2);

%% Random midpoint displacement
H = 0.8; g0 = 0.01;
rnd = 5.468815192049838e-01;
Z = ranmid(log2(mx),H,1,1,rnd);
Z = g0*(Z-min(min(Z)))/(max(max(Z))-min(min(Z)));

%% Export to .dat file, red by FEAP
DATA = [X(:) Y(:) Z(:)];
cd ..
  save(strcat('surf',string(mx),'.dat'),'DATA','-ascii');
cd pre_proc

%% Plots (full page picture: 478x318 pt^2 -> 6.6389x4.4167 in^2)
normal = [0 0 2.2083 1.6562]; % 8/6
big = [0 0 4.4167 4.4167];
full = [0 0 6.6389 4.4167];
landscape = [0 0 4.4167 1.6562];

size_fig = 2*normal;

error('### End of file ###')

figure(1); fig=gcf; ax=gca;
fig.Units='inches';
fig.Position=size_fig;
fig.Renderer='painters';                 %vectorial image, unclear
fig.PaperPosition=size_fig;
fig.PaperSize=size_fig(3:4);
fig.PaperOrientation='portrait';
surf(X,Y,Z/g0,'EdgeColor','none','LineStyle','none','FaceLighting','phong')
xlabel('$x/L$')
ylabel('$y/L$')
zlabel('$z/g_0$')
grid on
colormap spring
ax.FontSize=9;
ax.PlotBoxAspectRatio = [1 1 .1]
% print(hf,'-dpdf','Test_plot.pdf', '-opengl')
% set(fig, 'Renderer', 'opengl') 
saveas(gcf,'RMD_vect','pdf')

% figure(2); fig=gcf; ax=gca;
% fig.Units='inches';
% fig.Position=[0.0 0.0 6.639 4.4167];
% fig.Renderer='painters';                 %vectorial image, unclear
% fig.PaperPosition=[0.0 0.0 6.639 4.4167];
% fig.Papersize_fig=[6.639 4.4167];
% fig.PaperOrientation='portrait';
% surf(X,Y,Z)
% xlabel('$x$')
% ylabel('$y$')
% zlabel('$z$')
% grid on
% colormap jet
% ax.Fontsize_fig=9;
% saveas(gcf,'surfl','pdf')
