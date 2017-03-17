function [] = TrainHMM()

% Number of sequences:
seqCount = 5;

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

% Number of distinct classes:
Z = 3;

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

display(Pi);
display(Theta);
for i = 1:Z
  display(strcat('Gaussian Data for class', int2str(i)));
  display(Mean{i});
  display(Var{i});
end
