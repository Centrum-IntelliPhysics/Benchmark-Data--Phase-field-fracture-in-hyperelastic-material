N = 100; % Number of realizations
gamma = 4; tau = 5; sigma = 25^2; % GRF parameters
X = linspace(-1.1, 1.1, 201); % Coarse grid on [-1.1, 1.1]
mean = 1;

for j = 1:N
    u0 = GRF_fenics(100, mean, gamma, tau, sigma, "periodic", [-1.1, 1.1]);
    bc_data(j, :) = u0(X); % Evaluate on the grid
end

save('bc_displacement.mat', 'bc_data', 'X', 'gamma', 'tau', 'sigma');

function u = GRF_fenics(N, m, gamma, tau, sigma, type, domain)

    if type == "dirichlet"
        m = 0;
    end
    
    if type == "periodic"
        my_const = 2*pi;
    else
        my_const = pi;
    end
    
    my_eigs = sqrt(2)*(abs(sigma).*((my_const.*(1:N)').^2 + tau^2).^(-gamma/2));
    
    if type == "dirichlet"
        alpha = zeros(N,1);
    else
        xi_alpha = randn(N,1);
        alpha = my_eigs.*xi_alpha;
    end
    
    if type == "neumann"
        beta = zeros(N,1);
    else
        xi_beta = randn(N,1);
        beta = my_eigs.*xi_beta;
    end
    
    a = alpha/2;
    b = -beta/2;
    
    c = [flipud(a) - flipud(b).*1i;m + 0*1i;a + b.*1i];
    
    if type == "periodic"
        uu = chebfun(c, domain, 'trig', 'coeffs'); % Use domain argument here
        u = chebfun(@(t) uu((t - (domain(1) + domain(2)) / 2)), domain, 'trig');
    else
        uu = chebfun(c, domain, 'trig', 'coeffs');
        u = chebfun(@(t) uu(t * (domain(2) - domain(1)) + domain(1)), domain);
    end
end