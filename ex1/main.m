data = load('ex1data1.txt'); % read comma separated data
X = data(:, 1); y = data(:, 2);
plotData(X,y)

m = length(X); % number of training examples
X = [ones(m,1),data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
J = computeCost(X,y,theta); % compute cost function
%fprintf('J = %f\n',J)

% Run gradient descent:
% Compute theta
iterations = 1500;
alpha = 0.01;
theta = gradientDescent(X, y, theta, alpha, iterations);

% Print theta to screen
% Display gradient descent's result
%fprintf('Theta computed from gradient descent:\n%f,\n%f\n',theta)

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-','LineWidth',1.5,'Color','Blue')
legend('Training data', 'Linear regression')
hold off

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
%fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);

% Visualizing J(theta_0, theta_1):
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20),'LineWidth',1.5)
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;



% Comparison between different methods to compute theta
% Using feature normalization 

fprintf('\n\nSingle Variable')
fprintf('\n\n---------------\n')

% Scale features and set them to zero mean
fprintf('\n\nfeature scaling of original data\n')
[X_norm, mu, sigma] = featureNormalize(data(:,1));
fprintf('-> mean of X_norm = %f\n',mean(X_norm))
fprintf('-> std of X_norm = %f\n',std(X_norm))

y = data(:,2);
X = [ones(m,1),X_norm]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

iterations = 1500;
alpha = 0.01;
theta_GD = gradientDescent(X, y, theta, alpha, iterations);
theta_NE = normalEqn(X,y);

fprintf('-> theta from GD = [%f %f]\n',theta_GD)
fprintf('---> J(theta) = %f\n',computeCost(X,y,theta_GD))
predict1 = [1, 3.5] *theta_GD;
fprintf('---> For population = 35,000, we predict a profit of %f\n\n', predict1*10000);
fprintf('-> theta from NE = [%f %f]\n',theta_NE)
fprintf('---> J(theta) = %f\n',computeCost(X,y,theta_NE))
predict1 = [1, 3.5] *theta_NE;
fprintf('---> For population = 35,000, we predict a profit of %f\n\n', predict1*10000);


fprintf('\n\nMultiple Variables')
fprintf('\n\n---------------\n')

% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
% First 10 examples from the dataset
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fprintf('\n\nfeature scaling of original data\n')
[X_norm, mu, sigma] = featureNormalize(data(:,1));
fprintf('-> mean of X_norm = %f\n',mean(X_norm))
fprintf('-> std of X_norm = %f\n',std(X_norm))

y = data(:,2);
X = [ones(m,1),X_norm]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

iterations = 1500;
alpha = 0.01;
theta_GD = gradientDescent(X, y, theta, alpha, iterations);
theta_NE = normalEqn(X,y);

fprintf('-> theta from GD = [%f %f]\n',theta_GD)
fprintf('---> J(theta) = %f\n',computeCost(X,y,theta_GD))
predict1 = [1, 3.5] *theta_GD;
fprintf('---> For population = 35,000, we predict a profit of %f\n\n', predict1*10000);
fprintf('-> theta from NE = [%f %f]\n',theta_NE)
fprintf('---> J(theta) = %f\n',computeCost(X,y,theta_NE))
predict1 = [1, 3.5] *theta_NE;
fprintf('---> For population = 35,000, we predict a profit of %f\n\n', predict1*10000);
