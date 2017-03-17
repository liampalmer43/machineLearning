function [] = TrainMixtureOfGaussians()

% Number of classes:
Z = 3;
% 10 Data Blocks
D = cell(5, 1);
% 10 Label Blocks
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
display(Pi);
for i = 1:Z
  display(strcat('Mean for class', int2str(i)));
  display(M{i});
end
display(SIG);

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
