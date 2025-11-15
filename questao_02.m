clc; clear; close all;

base = [0 0 -1
        0 1 -1
        1 0 -1
        1 1 -1]

d = [0 1 1 1]
n = 0.2
w = [0 0 1]


for epoca = 1:10
  for i = 1:size(base,1)
    x = base(i,:);
    u = x * w';
    y = u >= 0;
    e = d(i) - y;
    w = w + n * e * x;
  end
end



x1 = base(:,1);
x2 = base(:,2);

figure;
hold on;
grid on;

plot(x1(d==0), x2(d==0), 'ro', 'MarkerFaceColor','r');
plot(x1(d==1), x2(d==1), 'bo', 'MarkerFaceColor','b');

xlabel('x1');
ylabel('x2');
title('Classificação Perceptron - Porta OR');


x_vals = linspace(-0.2, 1.2, 100);
y_vals = (w(3) - w(1)*x_vals) / w(2);

plot(x_vals, y_vals, 'k-', 'LineWidth', 2);
legend('Classe 0','Classe 1','Fronteira de decisão');
hold off;


