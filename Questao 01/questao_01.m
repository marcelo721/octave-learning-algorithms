hold on;

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


X_polinomial = [ones(rows(base),1), base(:,1:2), base(:,1:2).^2, base(:,1:2).^3,base(:,1:2).^4];

y = base(:,3);

betas = (X_polinomial'*X_polinomial)^-1*X_polinomial'*y


[x1_grid, x2_grid] = meshgrid(linspace(min(base(:,1)), max(base(:,1)), 30),linspace(min(base(:,2)), max(base(:,2)), 30));

y_grid = betas(1) + betas(2)*x1_grid + betas(3)*x2_grid + betas(4)*x1_grid.^2 + betas(5)*x2_grid.^2 + betas(6)*x1_grid.^3 + betas(7)*x2_grid.^3 + betas(8)*x1_grid.^4 + betas(9)*x2_grid.^4;
surf(x1_grid, x2_grid, y_grid, 'FaceAlpha', 0.5);

plot3(base(:,1), base(:,2), y, 'o', 'MarkerSize', 7,
      'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r');

y_pred = X_polinomial * betas;
R2 = 1 - sum((y - y_pred).^2) / sum((y - mean(y)).^2);
disp(['R² = ', num2str(R2)]);

R2_text = ['R² = ', num2str(R2)];

annotation('textbox', [0.70, 0.80, 0.25, 0.10], 'String', R2_text,
           'FontSize', 20,
           'FontWeight', 'bold',
           'HorizontalAlignment', 'right',
           'VerticalAlignment', 'top',
           'BackgroundColor', 'white',
           'EdgeColor', 'black');

