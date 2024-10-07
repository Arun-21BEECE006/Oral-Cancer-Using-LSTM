function Inf=normalizebm(Inf)
% Normalize to the range of [0,1]
 
fmin  = min(Inf(:));
fmax  = max(Inf(:));
Inf = (Inf-fmin)/(fmax-fmin);  % Normalize f to the range [0,1]