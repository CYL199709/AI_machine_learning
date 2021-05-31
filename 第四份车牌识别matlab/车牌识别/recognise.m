function ch = recognise(Theta1, Theta2, X, lib,i)

% �����Ѿ�ѵ���õ�������Ĳ���Theta1��Theta2����X����ʶ��

if i == 0
    m = size(X, 1);

    h1 = sigmoid([ones(m, 1) X] * Theta1');
    h2 = sigmoid([ones(m, 1) h1] * Theta2');
    [~, p] = max(h2, [], 2);

    ch = lib(15);
end
if i == 1
    m = size(X, 1);

    h1 = sigmoid([ones(m, 1) X] * Theta1');
    h2 = sigmoid([ones(m, 1) h1] * Theta2');
    [~, p] = max(h2, [], 2);

    ch = lib(p);
end
end
