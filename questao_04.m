
clear all; close all; clc;

base = load("two_classes.dat");

X = base(:,1:2);
X = normalize(X);
y = base(:,3);

n = 6
c = rand(n,2)


d = zeros(rows(X), n);
sigma = 1

Phi = zeros(rows(X), n);

for i=1:rows(X)
  for j=1:n
    d(i,j) = norm(X(i,:) - c(j,:));
  end
end

for j = 1:n
  Phi(:,j) = exp(-(d(:,j).^2) / (2*sigma^2));
end

w = (Phi' * Phi)^-1 * Phi' * y;

% ===============================================================
% 6. Criar grade para plotar superfície de decisão
% ===============================================================
x1 = linspace(min(X(:,1)), max(X(:,1)), 100);
x2 = linspace(min(X(:,2)), max(X(:,2)), 100);

[X1grid, X2grid] = meshgrid(x1, x2);

% matriz para armazenar ativações
PhiGrid = zeros(numel(X1grid), n);

% ===============================================================
% 7. Calcular ativações da RBF na grade 2D
% ===============================================================
for j = 1:n
    dist = sqrt((X1grid(:) - c(j,1)).^2 + (X2grid(:) - c(j,2)).^2);
    PhiGrid(:,j) = exp(-(dist.^2) / (2*sigma^2));
end

% ===============================================================
% 8. Predição da rede na grade
% ===============================================================
Ygrid = PhiGrid * w;

% Converter para forma matricial para plot
Ygrid = reshape(Ygrid, size(X1grid));

% ===============================================================
% 9. Plotar superfície de decisão
% ===============================================================
figure;

% Colormap muito mais bonito
colormap(viridis());

% Fundo suave com a saída da RBF
contourf(X1grid, X2grid, Ygrid, 50, "LineColor", "none");
hold on;

% Linha de decisão
contour(X1grid, X2grid, Ygrid, [0 0], "LineWidth", 3, "LineColor", "k");

% Plotar classes com cores mais profissionais
idx1 = find(y == 1);
idx2 = find(y == -1);

scatter(X(idx1,1), X(idx1,2), 35, [0.1 0.4 0.9], "filled");   % azul suave
scatter(X(idx2,1), X(idx2,2), 35, [0.9 0.2 0.1], "filled");   % vermelho suave

title("Superfície de Decisão da Rede RBF");
xlabel("X1");
ylabel("X2");

colorbar;



