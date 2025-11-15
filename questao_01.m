hold on
base = [122 139 0.115;
 114 126 0.120;
 086 090 0.105;
 134 144 0.090;
 146 163 0.100;
 107 136 0.120;
 068 061 0.105;
 117 062 0.080;
 071 041 0.100;
 098 120 0.11]

inputs_linear = base(:,1:2)
inputs_quadratico = inputs_linear.^2
X = [ones(size(inputs_linear, 1), 1), inputs_linear]

X_polinomial = [X, inputs_quadratico]
inputs_cubico = inputs_linear.^3;
X_polinomial = [X_polinomial, inputs_cubico];
y = base(:,3)
betas = (X_polinomial'*X_polinomial)^-1*X_polinomial'*y


[x1_grid, x2_grid] = meshgrid(linspace(min(base(:,1)), max(base(:,1)), 30), ...
                              linspace(min(base(:,2)), max(base(:,2)), 30));

y_grid = betas(1) + betas(2)*x1_grid + betas(3)*x2_grid + betas(4)*x1_grid.^2 + betas(5)*x2_grid.^2 + betas(6)*x1_grid.^3 + betas(7)*x2_grid.^3;
surf(x1_grid, x2_grid, y_grid, 'FaceAlpha', 0.5);

plot3(base(:,1),base(:,2) , y, 'o',4);

y_pred = X_polinomial * betas;
R2 = 1 - sum((y - y_pred).^2) / sum((y - mean(y)).^2);
disp(['R² = ', num2str(R2)]);


