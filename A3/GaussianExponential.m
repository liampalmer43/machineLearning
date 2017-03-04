function [] = gaussianExponential()

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

% Calculate Euclidean loss for different sigmas:
for s = 1:6
  x(s) = s;
  y(s) = crossValidation(D, L, s);
  display(strcat('sigma-', num2str(s)));
  display(strcat('avgLoss-', num2str(y(s),'%.8f')));
end

display(x);
display(y);
% Plot results.
figure(1);
plot(x, y, 'b');
title('Euclidean Loss vs sigma');
xlabel('sigma');
ylabel('Euclidean Loss');
hold on;

% Returns the average Euclidean loss of 10-fold cross validation on Gaussian process regression (exponential).
function avgLoss = crossValidation(D, L, sig)
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
    loss = applyGaussianExponential(data, labels, testData, testLabels, sig);
    result = result + loss;
  end
  avgLoss = result / 10;

% Returns the loss of exponential Gaussian Process.
% Training Set: data & labels.
% Testing Set: testData & testLabels.
function loss = applyGaussianExponential(data, labels, testData, testLabels, sig)
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
      K(i,j) = K(i,j) + kernel(data(i,:), data(j,:), sig);
    end
  end
  C = inv(K)*labels;

  % For each test:
  for i = 1:testCount
    ker = zeros(1, dataCount);
    for j = 1:dataCount
      ker(1,j) = kernel(testData(i,:), data(j,:), sig);
    end
    y = ker*C;
    % Tally the Euclidean Lose.
    totalLoss = totalLoss + (testLabels(i,1) - y)^2;
  end

  loss = totalLoss / testCount;

% Compute the kernel function
function r = kernel(v1, v2, sig)
  % row vectors
  r = exp(-norm(v1-v2)^2/(2*sig*sig));
