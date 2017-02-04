function [] = logisticRegression()

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
w = getParameters(data, labels);
display(w);

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
    acc = applyLR(data, labels, testData, testLabels);
    result = result + acc;
  end
  avgAccuracy = result / 10;

% Returns the accuracy of Mixture of Gaussians.
% Training Set: data & labels.
% Testing Set: testData & testLabels.
function accuracy = applyLR(data, labels, testData, testLabels)
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
 
  w = getParameters(data, labels);
  [d1, d2] = size(w);
  assert(d1 == n+1 && d2 == 1);
 
  correct = 0;
  incorrect = 0;
  % For each test:
  for i = 1:testCount
    [d1, d2] = size(testData(i,:));
    assert(d1 == 1 && d2 == 64);
    prob5 = sig(transpose(w)*transpose([1 testData(i,:)]));
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

function w = getParameters(data, labels)
  [m,n] = size(data);
  dataCount = m;

  % Logistic regression parameter estimation.
  w = zeros(n+1, 1);

  for iterations = 1:10
    % Calculate the Hessian.
    H = zeros(n+1, n+1);
    gradL = zeros(n+1, 1);
    for i = 1:dataCount
      d = [1 ; transpose(data(i,:))];
      [d1, d2] = size(d);
      assert(d1 == n+1 && d2 == 1);
      v = sig(transpose(w)*d);
      H = H + v*(1-v)*d*transpose(d);
      label = labels(i,1);
      assert(label == 5 || label == 6);
      y = tern(label == 5, 1, 0); 
      gradL = gradL + (v-y)*d;
    end
    w = w - inv(H)*gradL;
  end

% Sigmoid function.
function s = sig(n)
  s = 1 / (1 + exp(-n));

% Ternary statement.
function r = tern(b, r1, r2)
  if (b)
    r = r1;
  else
    r = r2;
  end
