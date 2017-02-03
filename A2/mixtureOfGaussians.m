function [] = knn()

% 10 Data Blocks
D = cell(10, 1);
% 10 Label Blocks
L = cell(10, 1);
% Initialize blocks.
for i = 1:10
  dataFileName = strcat('mixtureOfGaussians/data', strcat(int2str(i), '.csv'));
  labelsFileName = strcat('mixtureOfGaussians/labels', strcat(int2str(i), '.csv'));
  D{i} = csvread(dataFileName);
  L{i} = csvread(labelsFileName);
  [a,b] = size(D{i});
  [c,d] = size(L{i});
  assert(a == 111 && c == 111 && b == 64 && d == 1);
end

% Calculate average accuracy.
result = crossValidation(D, L);
display(result);

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
  [a,b] = size(testData);
  [c,d] = size(testLabels);
  assert(m == p && m == 999);
  assert(a == c && a == 111);
  assert(n == b && n == 64);
  assert(q == d && q == 1);
  testCount = a;
  dataCount = m;
  correct = 0;
  incorrect = 0;

  % Maximum likelihood parameter estimation.
  total = 0;
  total5 = 0;
  total6 = 0;
  total5Data = zeros(1, n);
  total6Data = zeros(1, n);
  % For each training:
  for i = 1:dataCount
    label = labels(i, 1);
    assert(label == 5 || label == 6);
    if label == 5
      total5 = total5 + 1;
      total5Data = total5Data + data(i,:);
    else
      total6 = total6 + 1;
      total6Data = total6Data + data(i,:);
    end
    total = total + 1;
  end
  avg5 = total5Data / total5;
  avg6 = total6Data / total6;
  prob5 = total5 / total;
  prob6 = total6 / total;
  assert(abs(prob5+prob6-1) < 0.000001);
  
  sum5 = 0;
  sum6 = 0;
  for i = 1:dataCount
    label = labels(i, 1);
    assert(label == 5 || label == 6);
    if label == 5
      sum5 = sum5 + (data(i,:)*transpose(data(i,:)));
    else
      sum6 = sum6 + (data(i,:)*transpose(data(i,:)));
    end
  end
  variance = (sum5 + sum6) / total;

  % Estimation parameters.
  w = (avg5 - avg6) / variance;
  [d1,d2] = size(w);
  assert(d1 == 1 && d2 == n);
  w0 = -0.5*avg5*transpose(avg5)/variance + 0.5*avg6*transpose(avg6)/variance + log(prob5/prob6);

  % For each test:
  for i = 1:testCount
    chance5 = sig(w*transpose(testData(i,:)) + w0);
    if chance5 >= 0.5 && testLabels(i, 1) == 5
      correct = correct + 1;
    elseif chance5 < 0.5 && testLabels(i, 1) == 6
      correct = correct + 1;
    else
      incorrect = incorrect + 1;
    end
  end

  accuracy = correct / (correct + incorrect);

% Sigmoid function.
function s = sig(n)
  s = 1 / (1 + exp(-n));
