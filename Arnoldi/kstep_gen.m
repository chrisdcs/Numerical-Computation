function [y, plam, lambda, residual]=kstep_gen(y,A,k)
% -- first, create an orthonormal Krylov basis (truncated Arnoldi)
dim = length(y);
Q = zeros(dim,k); H = zeros(k,k); 
Q(:,1) = y/norm(y);
for n = 1:k-1;
  qn = Q(:,n);
  v = A*qn;
  for j = 1:n
    qj = Q(:,j);
    hjn = qj'*v;
    v = v - hjn*qj;
    H(j,n) = hjn;
  end
  hn1n = norm(v);
  Q(:,n+1) = v/hn1n;
  H(n+1,n) = hn1n;
end
qk = Q(:,k);
v = A*qk;
for j = 1:k
  qj = Q(:,j); hjk = qj'*v;
  v = v - hjk*qj; H(j,k) = hjk;
end

[V,U] = eig(H);
d  = diag(U); ad = abs(d);
[S,II] = maxk(ad,2);
lambda = d(II(1)); plam = d(II(2)); 
ve = V(:,II(1));
y = Q*ve;
residual = norm(A*y - lambda*y);

