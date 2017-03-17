function [] = ApplyMixtureOfGaussians()

% Number of classes:
Z = 3;
% 5 Data Blocks
D = cell(5, 1);
% 5 Label Blocks
L = cell(5, 1);
% Initialize blocks.
for i = 1:5
  dataFileName = strcat('data/trainDataSeq', strcat(int2str(i), '.csv'));
  labelsFileName = strcat('data/trainLabelSeq', strcat(int2str(i), '.csv'));
  D{i} = csvread(dataFileName);
  L{i} = csvread(labelsFileName);
  [a,b] = size(D{i});
  [c,d] = size(L{i});
  assert(a == 100 && c == a && b == 2 && d == 1);
end

% Train on whole data set.
data = [];
labels = [];
for i = 1:5
  data = [data; D{i}];
  labels = [labels; L{i}];
end
[Pi, M, SIG] = getParameters(data, labels);

% Test on the test data set.
% 5 Test Data Blocks
TD = cell(5, 1);
% 5 Test Label Blocks
TL = cell(5, 1);
% Initialize blocks.
for i = 1:5
  dataFileName = strcat('data/testDataSeq', strcat(int2str(i), '.csv'));
  labelsFileName = strcat('data/testLabelSeq', strcat(int2str(i), '.csv'));
  TD{i} = csvread(dataFileName);
  TL{i} = csvread(labelsFileName);
  [a,b] = size(TD{i});
  [c,d] = size(TL{i});
  assert(a == 100 && c == a && b == 2 && d == 1);
end

right = 0;
wrong = 0;
for i = 1:5
  testData = TD{i};
  testLabels = TL{i};
  [a,b] = size(testData);
  [c,d] = size(testLabels);
  assert(a == 100 && c == a && b == 2 && d == 1);

  % The ith entry of PS is an array [ai, bi, ci...] with
  % ai = Pr(yi=a|x1...xi), bi = Pr(yi=b|x1...xi), etc.
  PS = cell(a, 1);
  
  for data = 1:a
    x = testData(data,:);
    p = zeros(Z,1);
    sumProb = 0;
    for class = 1:Z
      p(class,1) = Pi(class)*exp(-0.5*(x-M{class})*inv(SIG)*transpose(x-M{class}));
      sumProb = sumProb + p(class,1);
    end
    p = p / sumProb;
    PS{data} = p;
  end

  for data = 1:a
    [m,index] = max(PS{data});
    if index == testLabels(data,1)
      right = right + 1;
    else
      wrong = wrong + 1;
    end  
  end
end

display(right);
display(wrong);
display(right / (right+wrong));

function [Pi, M, SIG] = getParameters(data, labels)
  [m,n] = size(data);
  dataCount = m;

  % Maximum likelihood parameter estimation.

  % Number of classes:
  Z = 3;
  % Class Frequency:
  Pi = zeros(Z,1);
  % Class Counts:
  C = zeros(Z,1);
  % Total Count:
  T = 0;
  % Means:
  M = cell(Z,1);
  for i = 1:Z
    M{i} = zeros(1,n);
  end

  % For each training:
  for i = 1:dataCount
    [d1, d2] = size(data(i,:));
    assert(d1 == 1 && d2 == n);

    label = labels(i, 1);
    C(label,1) = C(label,1) + 1;
    M{label} = M{label} + data(i,:);
    T = T + 1;
  end

  for i = 1:Z
    M{i} = M{i} / C(i,1);
    Pi(i,1) = C(i,1) / T;
  end
  
  S = cell(Z,1);
  for i = 1:Z
    S{i} = zeros(n,n);
  end

  for i = 1:dataCount
    label = labels(i, 1);
    dist = data(i,:) - M{label};
    S{label} = S{label} + (transpose(dist)*dist);
  end

  SIG = zeros(n,n);
  for i = 1:Z;
    SIG = SIG + S{i};
  end
  SIG = SIG / T;
