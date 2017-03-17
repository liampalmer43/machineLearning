function [] = ForwardAlgorithmMonitoring()

% Number of sequences:
seqCount = 5;
% Number of classes:
Z = 3;

% Training data only:
% 5 Data Sequences
% 5 Label Sequences
D = cell(seqCount, 1);
L = cell(seqCount, 1);
for i = 1:seqCount
  dataFileName = strcat('data/trainDataSeq', strcat(int2str(i), '.csv'));
  labelsFileName = strcat('data/trainLabelSeq', strcat(int2str(i), '.csv'));
  D{i} = csvread(dataFileName);
  L{i} = csvread(labelsFileName);
  [a,b] = size(D{i});
  [c,d] = size(L{i});
  assert(a == 100 && a == c && b == 2 && d == 1);
end
[Pi,Theta,Mean,Var] = trainHMM(D, L);

% Testing data only:
% 5 Test Data Sequences
% 5 Test Label Sequences
TD = cell(seqCount, 1);
TL = cell(seqCount, 1);
for i = 1:seqCount
  dataFileName = strcat('data/testDataSeq', strcat(int2str(i), '.csv'));
  labelsFileName = strcat('data/testLabelSeq', strcat(int2str(i), '.csv'));
  TD{i} = csvread(dataFileName);
  TL{i} = csvread(labelsFileName);
  [a,b] = size(TD{i});
  [c,d] = size(TL{i});
  assert(a == 100 && a == c && b == 2 && d == 1);
end

right = 0;
wrong = 0;
for i = 1:seqCount
  testData = TD{i};
  testLabels = TL{i};
  [a,b] = size(TD{i});
  [c,d] = size(TL{i});
  assert(a == 100 && a == c && b == 2 && d == 1);

  % The ith entry of PS is an array [ai, bi, ci...] with
  % ai = Pr(yi=a|x1...xi), bi = Pr(yi=b|x1...xi), etc.
  PS = cell(a, 1);
  % Base Case:
  p1 = zeros(Z,1);
  sumProb = 0;
  for class = 1:Z
    p1(class,1) = GaussianProbability(testData(1,:), Mean{class}, Var{class}) * Pi(class);
    sumProb = sumProb + p1(class,1);
  end
  p1 = p1 / sumProb;
  PS{1} = p1; 

  for data = 2:a
    p = zeros(Z,1);
    sumProb = 0;
    for class = 1:Z
      pLast = PS{data-1};
      rec = 0;
      for r = 1:Z
        rec = rec + Theta(class,r)*pLast(r);
      end
      p(class,1) = GaussianProbability(testData(data,:), Mean{class}, Var{class}) * rec;
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

function p = GaussianProbability(x, u, v)
  p = exp(-0.5*(x-u)*inv(v)*transpose(x-u)) / (2*pi*sqrt(det(v)));

function [Pi,Theta,Mean,Var] = trainHMM(D, L)
  % Number of distinct classes:
  Z = 3;
  % Number of sequences:
  seqCount = 5;

  % Start counts:
  S = zeros(Z,1);
  % Transition counts:
  T = zeros(Z,Z);
  % Class counts:
  C = zeros(Z,1);
  % Data sums:
  U = cell(Z,1);
  for i = 1:Z
    U{i} = zeros(1,2);
  end

  % For each sequence:
  for s = 1:seqCount
    dataSeq = D{s};
    labelSeq = L{s};
    [a,b] = size(dataSeq);
    [c,d] = size(labelSeq);
    S(labelSeq(1,1), 1) = S(labelSeq(1,1), 1) + 1;
    for i = 1:a
      if i ~= a
        T(labelSeq(i+1,1), labelSeq(i,1)) = T(labelSeq(i+1,1), labelSeq(i,1)) + 1;
      end
      C(labelSeq(i,1),1) = C(labelSeq(i,1),1) + 1;
      U{labelSeq(i,1)} = U{labelSeq(i,1)} + dataSeq(i,:);
    end
  end

  % Parameters:
  Pi = zeros(Z,1);
  for i = 1:Z
    Pi(i,1) =  S(i,1) / seqCount;
  end

  Theta = zeros(Z,Z);
  for i = 1:Z
    for j = 1:Z
      div = 0;
      for k = 1:Z
        div = div + T(k,j);
      end
      Theta(i,j) = T(i,j) / div;
    end
  end

  Mean = cell(Z,1);
  for i = 1:Z
    Mean{i} = U{i} / C(i,1);
  end

  Var = cell(Z,1);
  V = cell(Z,1);
  for i = 1:Z
    V{i} = zeros(2,2);
  end
  for s = 1:seqCount
    dataSeq = D{s};
    labelSeq = L{s};
    [a,b] = size(dataSeq);
    [c,d] = size(labelSeq);
    for i = 1:a
      V{labelSeq(i,1)} = V{labelSeq(i,1)} + (transpose((dataSeq(i,:) - Mean{labelSeq(i,1)})) * (dataSeq(i,:) - Mean{labelSeq(i,1)}));
    end
  end
  for i = 1:Z
    Var{i} = V{i} / C(i,1);
  end
