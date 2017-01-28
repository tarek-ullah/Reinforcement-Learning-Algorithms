
function [v, pi] = policyIteration(model, maxit)
% initialize the value function
v = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);
% old_v = zeros(model.stateCount, 1);
policy = ones(model.stateCount, 1);
tol = 0.0000000000000000000001;
%run this extra loop
% to check for convergence
%in policy evaluation and
%policy improvement step
for iterations = 1:maxit,
% Policy Evaluation Step
%with same number of episodes
for i = 1:maxit,
v_ = zeros(model.stateCount, 1);
% perform the Bellman update for each state
for s = 1:model.stateCount
v_(s) = model.R(s, policy(s))
+ (model.gamma*model.P(s,:,policy(s))*v)’;
end
delta = norm(v - v_);
v = v_;
%check for convergence
if delta <= tol
fprintf(’Value function
converged after $\%$ d iterations\n’,i);
break;
end
end
for s = 1:model.stateCount
P =reshape(model.P(s,:,:),model.stateCount,4);
[~, action] =
max(model.R(s,:) + (model.gamma *P’*v)’);
policy(s) = action;
end
end
pi = policy;
v = v_;
end