function [] = mixtureOfGaussians()

% 10 Data Blocks
D = cell(10, 1);
% 10 Label Blocks
L = cell(10, 1);
% Initialize blocks.
for i = 1:10
  dataFileName = strcat('data/data', strcat(int2str(i), '.csv'));
  labelsFileName = strcat('data/labels', strcat(int2str(i), '.csv'));
  D{i} = csvread(dataFileName);
  L{i} = csvread(labelsFileName);
  [a,b] = size(D{i});
  [c,d] = size(L{i});
  assert(a == 111 && c == 111 && b == 64 && d == 1);
end

% Calculate average accuracy.
result = crossValidation(D, L);
display(result);

% Train on whole data set, show w and w0.
data = [];
labels = [];
for i = 1:10
  data = [data; D{i}];
  labels = [labels; L{i}];
end
[w, w0, pi5, pi6, u5, u6, SIG] = getParameters(data, labels);
display(w0);
display(w);
display(strcat('Pi1 = ', num2str(pi5)));
display(strcat('Pi2 = ', num2str(pi6)));
display(u5);
display(u6);
[d1, d2] = size(SIG);
assert(d1 == d2);
for i = 1:d1
  diag(i) = SIG(i,i);
end
display(diag);

% Returns the accuracy of 10-fold cross validation using Mixture of Gaussians.
function avgAccuracy = crossValidation(D, L)
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
    acc = applyMOG(data, labels, testData, testLabels);
    result = result + acc;
  end
  avgAccuracy = result / 10;

% Returns the accuracy of Mixture of Gaussians.
% Training Set: data & labels.
% Testing Set: testData & testLabels.
function accuracy = applyMOG(data, labels, testData, testLabels)
  [m,n] = size(data);
  [p,q] = size(labels);
  assert(m == 999);
  assert(p == 999);
  assert(n == 64);
  assert(q == 1);
  dataCount = m;
  [a,b] = size(testData);
  [c,d] = size(testLabels);
  assert(a == c && a == 111);
  assert(b == 64);
  assert(d == 1);
  testCount = a;
 
  [w, w0, pi5, pi6, u5, u6, SIG] = getParameters(data, labels);
  [d1, d2] = size(w);
  assert(d1 == 1 && d2 == 64);
 
  correct = 0;
  incorrect = 0;
  % For each test:
  for i = 1:testCount
    [d1, d2] = size(testData(i,:));
    assert(d1 == 1 && d2 == 64);
    prob5 = sig(w*transpose(testData(i,:)) + w0);
    assert(prob5 >= 0 && prob5 <= 1);
    label = testLabels(i, 1);
    assert(label == 5 || label == 6);

    if prob5 >= 0.5 && label == 5
      correct = correct + 1;
    elseif prob5 < 0.5 && label == 6
      correct = correct + 1;
    else
      incorrect = incorrect + 1;
    end
  end
  assert(correct + incorrect == testCount);

  accuracy = correct / (correct + incorrect);

function [w, w0, pi5, pi6, u5, u6, SIG] = getParameters(data, labels)
  [m,n] = size(data);
  dataCount = m;

  % Maximum likelihood parameter estimation.
  N = 0;
  N5 = 0;
  N6 = 0;
  u5 = zeros(1, n);
  u6 = zeros(1, n);
  % For each training:
  for i = 1:dataCount
    [d1, d2] = size(data(i,:));
    assert(d1 == 1 && d2 == n);
    label = labels(i, 1);
    assert(label == 5 || label == 6);
    if label == 5
      N5 = N5 + 1;
      u5 = u5 + data(i,:);
    else
      N6 = N6 + 1;
      u6 = u6 + data(i,:);
    end
    N = N + 1;
  end
  u5 = u5 / N5;
  u6 = u6 / N6;
  pi5 = N5 / N;
  pi6 = N6 / N;
  assert(abs(pi5+pi6-1) < 0.000001);
  
  S5 = zeros(n, n);
  S6 = zeros(n, n);
  for i = 1:dataCount
    label = labels(i, 1);
    assert(label == 5 || label == 6);
    if label == 5
      dist = data(i,:)-u5;
      S5 = S5 + (transpose(dist)*dist);
    else
      dist = data(i,:)-u6;
      S6 = S6 + (transpose(dist)*dist);
    end
  end
  SIG = (S5 + S6) / N;

  % Estimation parameters.
  w = (u5 - u6) * inv(SIG);
  [d1,d2] = size(w);
  assert(d1 == 1 && d2 == n);
  w0 = -0.5*u5*inv(SIG)*transpose(u5) + 0.5*u6*inv(SIG)*transpose(u6) + log(pi5/pi6);

% Sigmoid function.
function s = sig(n)
  s = 1 / (1 + exp(-n));
