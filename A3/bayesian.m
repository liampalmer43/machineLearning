function [] = bayesian()

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

% Calculate Euclidean loss for deg = 1, 2, 3, 4
l = 1;
for d = 1:4
  x(d) = d;
  y(d) = crossValidation(D, L, d);
  display(strcat('degree', num2str(d)));
  display(strcat('avgLoss', num2str(y(d),'%.8f')));
end

display(x);
display(y);
% Plot results.
figure(1);
plot(x, y, 'b');
title('Euclidean Loss vs degree');
xlabel('degree');
ylabel('Euclidean Loss');
hold on;

% Returns the average Euclidean loss of 10-fold cross validation on `l` weighted linear regression.
function avgLoss = crossValidation(D, L, deg)
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
    loss = applyBayesianRegression(data, labels, testData, testLabels, deg);
    result = result + loss;
  end
  avgLoss = result / 10;

% Returns the loss of `l=1` weighted non-linear regression.
% Training Set: data & labels.
% Testing Set: testData & testLabels.
function loss = applyBayesianRegression(data, labels, testData, testLabels, deg)
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

  dim = 0;
  if deg == 1
    dim = 3;
  elseif deg == 2
    dim = 6;
  elseif deg == 3
    dim = 10;
  elseif deg == 4
    dim = 15;
  end

  % For each training:
  A = zeros(dim, dim);
  b = zeros(dim, 1);
  for j = 1:dataCount
    A = A + transpose([1 nonlinear(data(j,:), deg)])*[1 nonlinear(data(j,:), deg)];
    b = b + transpose([1 nonlinear(data(j,:), deg)])*labels(j,1);
  end
  M = A + eye(dim);
  w = M\b;

  % For each test:
  for i = 1:testCount
    % Tally the Euclidean Lose.
    totalLoss = totalLoss + [(testLabels(i,1) - [1 nonlinear(testData(i,:), deg)]*w)^2];
  end

  loss = totalLoss / testCount;

% Map v to dual space with degree up to d
function r = nonlinear(v, d)
  r(1) = v(1);
  r(2) = v(2);
  if d > 1
    r(3) = v(1)^2;
    r(4) = v(1)*v(2);
    r(5) = v(2)^2;
  end
  if d > 2
    r(6) = v(1)^3;
    r(7) = v(1)^2 * v(2);
    r(8) = v(1) * v(2)^2;
    r(9) = v(2)^3;
  end
  if d > 3
    r(10) = v(1)^4;
    r(11) = v(1)^3 * v(2);
    r(12) = v(1)^2 * v(2)^2;
    r(13) = v(1) * v(2)^3;
    r(14) = v(2)^4;
  end
