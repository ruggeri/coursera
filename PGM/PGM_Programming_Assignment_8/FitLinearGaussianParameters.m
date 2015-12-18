function [Beta sigma] = FitLinearGaussianParameters(X, U)

% Estimate parameters of the linear Gaussian model:
% X|U ~ N(Beta(1)*U(1) + ... + Beta(n)*U(n) + Beta(n+1), sigma^2);

% Note that Matlab/Octave index from 1, we can't write Beta(0).
% So Beta(n+1) is essentially Beta(0) in the text book.

% X: (M x 1), the child variable, M examples
% U: (M x N), N parent variables, M examples
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

M = size(U,1);
N = size(U,2);

% collect expectations and solve the linear system
% A = [ E[U(1)],      E[U(2)],      ... , E[U(n)],      1     ; 
%       E[U(1)*U(1)], E[U(2)*U(1)], ... , E[U(n)*U(1)], E[U(1)];
%       ...         , ...         , ... , ...         , ...   ;
%       E[U(1)*U(n)], E[U(2)*U(n)], ... , E[U(n)*U(n)], E[U(n)] ]
U = [ones(M, 1), U];
A = [];
for i=1:size(U, 2)
  u = U(:, i);
  A(:, end+1) = mean(u .* U)';
end

% B = [ E[X]; E[X*U(1)]; ... ; E[X*U(n)] ]
B = mean(X .* U)';

% solve A*Beta = B
Beta = A\B;

% then compute sigma according to eq. (11) in PA description
sigma = 1;
sigma = std(X - U*Beta, 1);

% Shuffle Beta around to account for stupid-ass indexing.
Beta = [Beta(2:end); Beta(1)];
