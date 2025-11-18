
clc; clear; close all;


X = [0 0;
     0 1;
     1 0;
     1 1];

d = [0; 1; 1; 0];


n_entradas = 2;
n_ocultos  = 2;
n_saida    = 1;
n_epocas   = 10000;
n = 0.15;


W1 = randn(n_ocultos, n_entradas + 1);
W2 = randn(n_saida, n_ocultos + 1);


sigmoid = @(x) 1 ./ (1 + exp(-x));
dsigmoid = @(y) y .* (1 - y);


for epoca = 1:n_epocas
    for i = 1:size(X,1)
        entrada = [X(i,:) 1]';
        u_oculto = W1 * entrada;
        y_oculto = sigmoid(u_oculto);

        y_oculto_bias = [y_oculto; 1];
        u_saida = W2 * y_oculto_bias;
        y_saida = sigmoid(u_saida);

        erro = d(i) - y_saida;

        delta_saida = erro .* dsigmoid(y_saida);
        delta_oculto = dsigmoid(y_oculto) .* (W2(:,1:end-1)' * delta_saida);

        W2 = W2 + n * delta_saida * y_oculto_bias';
        W1 = W1 + n * delta_oculto * entrada';
    end
end
% -----------------------------
% Loop de teste manual
% -----------------------------
disp(' ');
disp('--- Teste manual (digite -1 para sair) ---');

while true
    x1 = input('Digite a primeira entrada (0 ou 1): ');
    if x1 == -1
        disp('Encerrando...');
        break;
    end

    x2 = input('Digite a segunda entrada (0 ou 1): ');
    if x2 == -1
        disp('Encerrando...');
        break;
    end

    entrada_usuario = [x1 x2 1]';

    y_oculto = sigmoid(W1 * entrada_usuario);
    y_saida = sigmoid(W2 * [y_oculto; 1]);

    fprintf('Resultado da rede: %.4f\n', y_saida);

    if y_saida >= 0.5
        fprintf('Saída aproximada: 1\n\n');
    else
        fprintf('Saída aproximada: 0\n\n');
    end
end

