function [] = gaussianPolynomial()

% 10 Data Blocks
D = cell(10, 1);
% 10 Label Blocks
L = cell(10, 1);
% Initialize blocks.
for i = 1:10
  dataFileName = strcat('data/fData', strcat(int2str(i), '.csv'));
  labelsFileName = strcat('data/fLabels', strcat(int2str(i), '.csv'));
  D{i} = csvread(dataFileName);
  L{i} = csvread(labelsFileName);
  [a,b] = size(D{i});
  [c,d] = size(L{i});
  assert(a == 20 && c == 20 && b == 2 && d == 1);
end

% Calculate Euclidean loss for different degrees:
for d = 1:4
  x(d) = d;
  y(d) = crossValidation(D, L, d);
  display(strcat('degree-', num2str(d)));
  display(strcat('avgLoss-', num2str(y(d),'%.8f')));
end

display(x);
display(y);
% Plot results.
figure(1);
plot(x, y, 'b');
title('Euclidean Loss vs degree: Gaussian Processes Polynomial');
xlabel('degree');
ylabel('Euclidean Loss');
hold on;

% Returns the average Euclidean loss of 10-fold cross validation on Gaussian process regression (polynomial).
function avgLoss = crossValidation(D, L, deg)
  assert(deg == 1 || deg == 2 || deg == 3 || deg== 4);
  result = 0;
  % Split data and labels in ten different ways.
  for i = 1:10
    data = [];
    labels = [];
    for j = 1:10
      if j ~= i
        data = [data; D{j}];
        labels = [labels; L{j}];
      end
    end
    testData = D{i};
    testLabels = L{i};
    loss = applyGaussianPolynomial(data, labels, testData, testLabels, deg);
if deg == 4
  display('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
  display(loss);
end
    result = result + loss;
  end
  avgLoss = result / 10;

% Returns the loss of polynomial Gaussian Process.
% Training Set: data & labels.
% Testing Set: testData & testLabels.
function loss = applyGaussianPolynomial(data, labels, testData, testLabels, deg)
  assert(deg == 1 || deg == 2 || deg == 3 || deg== 4);
  [m,n] = size(data);
  [p,q] = size(labels);
  [a,b] = size(testData);
  [c,d] = size(testLabels);
  assert(m == p && m == 180);
  assert(a == c && a == 20);
  assert(n == b && n == 2);
  assert(q == d && q == 1);
  testCount = a;
  dataCount = m;
  totalLoss = 0;

  % For each training:
  K = eye(dataCount);
  for i = 1:dataCount
    for j = 1:dataCount
      K(i,j) = K(i,j) + kernel(data(i,:), data(j,:), deg);
    end
  end
if deg == 4
  %display(inv(K));
end
  C = inv(K)*labels;
if deg == 4
%display('--------------');
%display(C);
end
  % For each test:
  for i = 1:testCount
    ker = zeros(1, dataCount);
    for j = 1:dataCount
      ker(1,j) = kernel(testData(i,:), data(j,:), deg);
    end
    y = ker*C;
    % Tally the Euclidean Lose.
    totalLoss = totalLoss + (testLabels(i,1) - y)^2;
  end

  loss = totalLoss / testCount;

% Compute the kernel function
function r = kernel(v1, v2, deg)
  assert(deg == 1 || deg == 2 || deg == 3 || deg== 4);
  % row vectors
  r = (v1*transpose(v2) + 1)^deg;
