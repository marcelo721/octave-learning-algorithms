
clc; clear; close all;


X = [0 0;
     0 1;
     1 0;
     1 1];

d = [0; 1; 1; 0];


n_entradas = 2;
n_ocultos  = 2;
n_saida    = 1;
n_epocas   = 100000;
eta = 0.01;


W1 = rand(n_ocultos, n_entradas + 1) - 0.5;
W2 = rand(n_saida, n_ocultos + 1) - 0.5;


sigmoid = @(x) 1 ./ (1 + exp(-x));
dsigmoid = @(y) y .* (1 - y);


for epoca = 1:n_epocas
    erro_total = 0;

    for i = 1:size(X,1)
        % ---- Forward ----
        entrada = [X(i,:) 1]';             % adiciona bias
        u_oculto = W1 * entrada;
        y_oculto = sigmoid(u_oculto);

        y_oculto_b = [y_oculto; 1];        % adiciona bias na saída da oculta
        u_saida = W2 * y_oculto_b;
        y_saida = sigmoid(u_saida);

        % ---- Erro ----
        erro = d(i) - y_saida;
        erro_total = erro_total + erro^2;

        % ---- Backpropagation ----
        delta_saida = erro .* dsigmoid(y_saida);
        delta_oculto = dsigmoid(y_oculto) .* (W2(:,1:end-1)' * delta_saida);

        % ---- Atualização dos pesos ----
        W2 = W2 + eta * delta_saida * y_oculto_b';
        W1 = W1 + eta * delta_oculto * entrada';
    end

    % Exibe o erro médio por época
    if mod(epoca, 1000) == 0
        fprintf('Época %d, Erro médio: %.4f\n', epoca, erro_total/4);
    end
end

% -----------------------------
% Teste final automático
% -----------------------------
disp('--- Teste automático após o treinamento ---');
for i = 1:size(X,1)
    entrada = [X(i,:) 1]';
    y_oculto = sigmoid(W1 * entrada);
    y_saida = sigmoid(W2 * [y_oculto; 1]);
    fprintf('Entrada: [%d %d] -> Saída: %.4f\n', X(i,1), X(i,2), y_saida);
end

% -----------------------------
% Entrada do usuário
% -----------------------------
disp(' ');
disp('--- Teste manual ---');
x1 = input('Digite a primeira entrada (0 ou 1): ');
x2 = input('Digite a segunda entrada (0 ou 1): ');

entrada_usuario = [x1 x2 1]';
y_oculto = sigmoid(W1 * entrada_usuario);
y_saida = sigmoid(W2 * [y_oculto; 1]);

fprintf('Resultado da rede: %.4f\n', y_saida);

if y_saida >= 0.5
    fprintf('Saída aproximada: 1\n');
else
    fprintf('Saída aproximada: 0\n');
end

