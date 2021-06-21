function Z = WM2d(X,Y,G,lam0,D,gam,nw,M,phi)
    [my,mx] = size(X);
    Z = zeros(my,mx);
        
    A = lam0*(G/lam0)^(D-2)*sqrt(log(gam)/M);
    for m=1:M
        for n=1:nw
                Z = Z + ...
                A*gam^((D-3)*(n-1))*(cos(phi(m,n))-...
                                     cos((2*pi*gam^(n-1))/lam0*(X*cos(pi*m/M)+Y*sin(pi*m/M))+phi(m,n)));
        end
    end
end
