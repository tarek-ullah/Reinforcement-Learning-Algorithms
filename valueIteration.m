
% run VI on GridWorld
gridworld;
[v, pi] = valueIteration(model, 1000)
plotVP(v,pi, paramSet)
%value iteration algorithm
function [v, pi] = valueIteration(model, maxit)
% initialize the value function
v = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);
old_v = zeros(model.stateCount, 1);
threshold = 1.0000e-22;
for iterations = 1:maxit,
% initialize the policy and the new value function
policy = ones(model.stateCount, 1);
v_ = zeros(model.stateCount, 1);
% perform the Bellman update for each state
for s = 1:model.stateCount,
%compute transition probability
P = reshape(model.P(s,:,:), model.stateCount, 4);
%update value function

[v_(s,:), action] = max(model.R(s,:) +
(model.gamma * P’ * v)’);
%policy evaluated every step
policy(s,:) = action;
end
old_v = v;
v = v_;
pi = policy;
%break condition
%to check convergence of VI algorithm
if v - old_v <= threshold
fprintf(’Value function converged
after \%d iterations\n’,iterations);
break;
end
end
end