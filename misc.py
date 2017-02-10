import numpy as np
import tensorflow as tf

class RunningAvg():
    def __init__(self):
        self._sum = 0.0
        self._count = 1e-5

    def add(self, arr):
        self._sum += sum(arr)
        self._count += len(arr)

    def val(self):
        return self._sum / self._count

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def discounted_cumsum(arr, gamma):
    discounted = arr[-1:]
    for i in reversed(range(len(arr) - 1)):
        discounted.append(arr[i]+ discounted[-1] * gamma)
    return discounted[::-1]

def normalize(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-10)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def evaluate_policy(env, agent, itr=20):
    rewards = []
    for i in range(itr):
        _, _, rews = policy_rollout(env, agent)
        rewards.append(sum(rews))
    return np.mean(rewards)

def policy_rollout(env, agent, horizon=None, show=False):
    """Run one episode."""
    if horizon is None:
        if hasattr(env, "horizon"):
            horizon = env.horizon
        else:
            horizon = np.inf
    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []
    eps_length = 0
    while not done and (eps_length < horizon):
        obs.append(observation)
        action = agent.act(observation)

        observation, reward, done, _ = env.step(action)
        if show: env.render()
        acts.append(action)
        rews.append(reward)
        eps_length += 1
    return obs, acts, rews


def policy_continue(env, agent, steps, horizon=None,  show=False):
    """Run few steps - assumes env object is stateful"""
    if horizon is None:
        if hasattr(env, "horizon"):
            horizon = env.horizon
        else:
            horizon = np.inf

    reward, done = 0, False
    obs, acts, rews = [], [], []
    if hasattr(env, "state"):
        observation = env.state
    else:
        print "Resetting..."
        observation = env.reset()


    for i in range(steps):
        obs.append(observation)
        action = agent.act(observation)
        # if action == 2:

        observation, reward, done, _ = env.step(action)
        if show: env.render()

        acts.append(action)
        rews.append(reward)        
        if done:
            observation = env.reset()
            break
    return obs, acts, rews, done
