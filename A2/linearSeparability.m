function [] = linearSeparability()

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

% Train on whole data set.
data = [];
labels = [];
for i = 1:10
  data = [data; D{i}];
  labels = [labels; L{i}];
end
w = getParameters(data, labels);

% Test on whole data set.
[m,n] = size(data);
correct = 0;
incorrect = 0;
for i = 1:m
  [d1, d2] = size(data(i,:));
  assert(d1 == 1 && d2 == 64);
  prob5 = sig(transpose(w)*transpose([1 data(i,:)]));
  assert(prob5 >= 0 && prob5 <= 1);
  label = labels(i, 1);
  assert(label == 5 || label == 6);

  if prob5 >= 0.5 && label == 5
    correct = correct + 1;
  elseif prob5 < 0.5 && label == 6
    correct = correct + 1;
  else
    incorrect = incorrect + 1;
  end
end
assert(correct + incorrect == m);

display(correct);
display(incorrect);
display(correct / (correct+incorrect));

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
