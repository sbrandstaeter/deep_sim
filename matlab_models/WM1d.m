function z = WM1d(x,nw,D,gam,lam0,g0,phi)
    mx = length(x);
    z = zeros(mx,1);
    for i=1:mx
        for j = 1:nw+1
            z(i) = z(i) + g0*gam^((D-2)*(j-1))*cos(2*pi*gam^(j-1)*(x(i)-phi*max(x))/lam0);
        end
    end
    z = z-min(z);    
end
