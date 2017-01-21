function [] = regression()

% 10 Data Blocks
D = cell(10, 1);
% 10 Label Blocks
L = cell(10, 1);
% Initialize blocks.
for i = 1:10
  dataFileName = strcat('regressionData/fData', strcat(int2str(i), '.csv'));
  labelsFileName = strcat('regressionData/fLabels', strcat(int2str(i), '.csv'));
  D{i} = csvread(dataFileName);
  L{i} = csvread(labelsFileName);
  [a,b] = size(D{i});
  [c,d] = size(L{i});
  assert(a == 20 && c == 20 && b == 2 && d == 1);
end

% Calculate Euclidean loss for l = 0,0.1,0.2, ... ,3.9,4.0
l = 0;
index = 1;
while l <= 4.01
  x(index) = l;
  y(index) = crossValidation(D, L, l);
  display(strcat('lambda', num2str(l,'%.8f')));
  display(strcat('avgLoss', num2str(y(index),'%.8f')));
  l = l + 0.1;
  index = index + 1;
end

display(x);
display(y);
% Plot results.
figure(1);
plot(x, y, 'b');
title('Euclidean Loss vs lambda');
xlabel('lambda');
ylabel('Euclidean Loss');
hold on;

% Returns the average Euclidean loss of 10-fold cross validation on `l` weighted linear regression.
function avgLoss = crossValidation(D, L, l)
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
    loss = applyRegression(data, labels, testData, testLabels, l);
    result = result + loss;
  end
  avgLoss = result / 10;

% Returns the loss of `l` weighted linear regression.
% Training Set: data & labels.
% Testing Set: testData & testLabels.
function loss = applyRegression(data, labels, testData, testLabels, l)
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

  % For each test:
  for i = 1:testCount
    % For each training:
    A = zeros(3, 3);
    b = zeros(3, 1);
    for j = 1:dataCount
      A = A + transpose([1 data(j,:)])*[1 data(j,:)];
      b = b + transpose([1 data(j,:)])*labels(j,1);
    end
    M = A + 2*l*eye(3);
    w = M\b;

    % Tally the Euclidean Lose.
    totalLoss = totalLoss + [(testLabels(i,1) - [1 testData(i,:)]*w)^2]/2;
  end

  loss = totalLoss / testCount;
