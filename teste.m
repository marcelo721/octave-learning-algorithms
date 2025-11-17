clear all; close all; clc;

% ===============================
% 1. CARREGAR BASE
% ===============================
base = load("two_classes.dat");
X = base(:,1:2);
y = base(:,3);   % -1 ou +1
N = rows(X);

% ===============================
% 2. NORMALIZAR X
% ===============================
mu = mean(X);            % média de cada coluna
sigmaX = std(X);         % desvio-padrão de cada coluna
Xn = (X - mu) ./ sigmaX; % normalizado (Z-score)

% ===============================
% 3. DEFINIR CENTROS E SIGMA
% ===============================
n = 20;               % número de RBFs
c = randn(n,2);      % centros (na escala dos dados normalizados)

sigma = 1;           % sigma fixo (pode mudar depois)

% ===============================
% 4. MATRIZ Phi
% ===============================
Phi = zeros(N,n);

for i = 1:N
  for j = 1:n
    dist_ij = norm(Xn(i,:) - c(j,:));
    Phi(i,j) = exp(-(dist_ij^2) / (2*sigma^2));
  end
end

% ===============================
% 5. TREINO (Regressão Linear)
% ===============================
lambda = 1e-3;   % regularização
w = (Phi' * Phi + lambda*eye(n)) \ (Phi' * y);

% ===============================
% 6. GRADE PARA SUPERFÍCIE DE DECISÃO
% ===============================
x1 = linspace(min(X(:,1))-1, max(X(:,1))+1, 200);
x2 = linspace(min(X(:,2))-1, max(X(:,2))+1, 200);
[Xgrid, Ygrid] = meshgrid(x1, x2);

% normalizar grade usando mesma média e desvio da base!
Xgrid_n = (Xgrid - mu(1)) / sigmaX(1);
Ygrid_n = (Ygrid - mu(2)) / sigmaX(2);

Z = zeros(size(Xgrid));

for i = 1:size(Xgrid,1)
  for j = 1:size(Xgrid,2)
    ponto = [Xgrid_n(i,j), Ygrid_n(i,j)];
    phi_pt = zeros(n,1);
    for k = 1:n
      phi_pt(k) = exp(-(norm(ponto - c(k,:))^2) / (2*sigma^2));
    end
    Z(i,j) = phi_pt' * w;
  end
end

% ===============================
% 7. PLOT DA SUPERFÍCIE DE DECISÃO
% ===============================
figure;
hold on;

% fronteira (Z = 0)
contour(Xgrid, Ygrid, Z, [0 0], "k", "LineWidth", 3);

% regiões coloridas
contourf(Xgrid, Ygrid, Z, 20, "LineStyle", "none");

% dados originais (não-normalizados!)
scatter(X(y==1,1),  X(y==1,2), 60, "r", "filled");
scatter(X(y==-1,1), X(y==-1,2), 60, "b", "filled");

% centros (também precisam ser “desnormalizados” para plot)
c_plot = c .* sigmaX + mu;
scatter(c_plot(:,1), c_plot(:,2), 120, "k", "filled", "MarkerFaceAlpha", 0.5);

title("Superfície de decisão - RBF com entrada normalizada");
xlabel("x₁");
ylabel("x₂");
colorbar;

hold off;

