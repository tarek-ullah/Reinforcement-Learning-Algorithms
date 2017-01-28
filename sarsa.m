



% sarsa on SmallWorld
smallworld;
[v, pi, ~] = sarsa(model, 1000, 1000);
plotVP(v,pi, paramSet)
%SARSA Algorithm
function [v, pi, Cum_Rwd] =
sarsa(model, maxit, maxeps)
% initialize the value function
Q = zeros(model.stateCount, 4);
pi = ones(model.stateCount, 1);
alpha = 1;
policy = ones(model.stateCount, 1);
Cum_Rwd = zeros(maxeps, 1);
for i = 1:maxeps,
%every time we reset the episode,
start at the given startState
%get Start State
s = model.startState;
%OR INITIALIZE ACTION ARBITRARILY
a = 1;
%% initialize the first action
%%greedily as well
%%a = epsilon_greedy_policy(Q(s,:));
%FOR EACH STEP OF EPISODE
for iter = 1:maxit,
p = 0;
r = rand;
for next_state = 1:model.stateCount,
p = p + model.P(s, next_state, a);
if r <= p,
break;
end
end
%TAKE ACTION, OBSERVE S’ AND R
s_ = next_state;
%get R with given a
reward = model.R(s,a);
%taking discounted cum rewards
Cum_Rwd(i) = Cum_Rwd(i) + model.gamma * reward;
%CHOOSE A’ FROM S’ USING GREEDY POLICY
a_ = epsilon_greedy_policy(Q(s_,:), iter);
alpha = 1/iter;
% IMPLEMENT THE UPDATE RULE FOR Q HERE.
Q(s,a) = Q(s,a) + alpha *
[reward + model.gamma * Q(s_, a_) - Q(s,a)];
s = s_;
a = a_;
[~, idx] = max(Q(s,:));
policy(s) = idx;
q = Q(:, idx);
if s == model.goalState
break;
end
end
end
pi = policy;
v = q;
end
The MATLAB code for the epsilon greedy policy is
given below:
function action = epsilon_greedy_policy(Q, iter)
all_actions = [1 2 3 4];
epsilon = 1/iter;
probability = rand();
if probability < (1 - epsilon)
[~, action] = max(Q);
else
action = all_actions(randi(length(all_actions)));
end
end








