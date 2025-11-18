clear all; close all; clc;

base = load("two_classes.dat");

X = base(:,1:2);
X = normalize(X);
y = base(:,3);

n = 20;

[~, c] = kmeans(X, n);

D = pdist(c);
media_dist = mean(D);
sigma = media_dist;

%sigma = 1;
%c = randn(n,2);

d = zeros(rows(X), n);
Phi = zeros(rows(X), n);

for i=1:rows(X)
  for j=1:n
    d(i,j) = norm(X(i,:) - c(j,:));
  end
end

for j = 1:n
  Phi(:,j) = exp(-(d(:,j).^2) / (2*sigma^2));
end

Phi = [Phi ones(rows(Phi),1)];

w = (Phi' * Phi)^-1 * Phi' * y;

x1 = linspace(min(X(:,1)), max(X(:,1)), 100);
x2 = linspace(min(X(:,2)), max(X(:,2)), 100);

[X1grid, X2grid] = meshgrid(x1, x2);

PhiGrid = zeros(numel(X1grid), n);

for j = 1:n
    dist = sqrt((X1grid(:) - c(j,1)).^2 + (X2grid(:) - c(j,2)).^2);
    PhiGrid(:,j) = exp(-(dist.^2) / (2*sigma^2));
end

PhiGrid = [PhiGrid ones(rows(PhiGrid),1)];

Ygrid = PhiGrid * w;
Ygrid = reshape(Ygrid, size(X1grid));

figure;
colormap(viridis());

contourf(X1grid, X2grid, Ygrid, 50, "LineColor", "none");
hold on;

contour(X1grid, X2grid, Ygrid, [0 0], "LineWidth", 3, "LineColor", "k");

idx1 = find(y == 1);
idx2 = find(y == -1);

scatter(X(idx1,1), X(idx1,2), 35, [0.1 0.4 0.9], "filled");
scatter(X(idx2,1), X(idx2,2), 35, [0.9 0.2 0.1], "filled");

title("Superfície de Decisão da Rede RBF");
xlabel("X1");
ylabel("X2");
colorbar;




