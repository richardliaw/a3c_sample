import gym
import numpy as np
import tensorflow as tf
from misc import *
import time
import ray
# from PGA_online import PolicyGradientAgent
from actorcritic import ActorCritic

NUM_WORKERS = 2
GAMMA = 0.95
ray.init(num_workers=NUM_WORKERS)

def env_init():
    return gym.make('CartPole-v0')

def env_reinit(env):
    return env

ray.env.env = ray.EnvironmentVariable(env_init, env_reinit)

def ac_init():
    env = ray.env.env
    print env.action_space.n
    hparams = {
            'input_size': env.observation_space.shape[0],
            'hidden_size': 64,
            'num_actions': env.action_space.n,
            'learning_rate': 0.001,
            'entropy_wt': 0.01
    }
    return ActorCritic(hparams)

def ac_reinit(actor_critic):
    return actor_critic

ray.env.actor_critic = ray.EnvironmentVariable(ac_init, ac_reinit)

@ray.remote
def a3c_rollout_grad(params, avg_rwd):
    GAMMA = 0.95
    env = ray.env.env
    actor_critic = ray.env.actor_critic
    steps = 10

    actor_critic.set_weights(params['weights'])
    obs, acts, rews, done = policy_continue(env, actor_critic, steps)

    estimated_values = actor_critic.get_value(obs).flatten()
    assert len(estimated_values.shape) == 1

    if done:
        cur_rwd = 0
    else:
        cur_rwd = estimated_values[-1]

    rewards = []
    for r in reversed(rews):
        cur_rwd = r + GAMMA * cur_rwd 
        rewards.insert(0, cur_rwd)
    rewards = np.asarray(rewards)
    # if any(estimated_values == np.nan):

    norm_advs = normalize(rewards - estimated_values)
    # print "Bootstrapped Obs Rewards: ", rewards
    # print "Estimated Values: ", estimated_values
    grads = actor_critic.compute_gradients(obs, acts, norm_advs, rewards)
    info = {"done": done, "obs": obs, "rews": rewards, "real_rwds": rews}
    # if any([(np.isnan(i)).any() for i in grads]):


    return grads, info

def train(u_itr=5000):
    actor_critic = ray.env.actor_critic
    env = ray.env.env
    remaining = []
    critic_loss = []
    rwds = []
    obs = []
    cur_itr = 0
    params = {}
    average_reward = RunningAvg()

    ## debug
    sq_loss = lambda x, y: sum((x - y)**2)

    while cur_itr < u_itr:
        params['weights'] = actor_critic.get_weights()
        # if any([(np.isnan(i)).any() for i in params['weights'].values()]):
        
        param_id = ray.put(params)

        jobs = NUM_WORKERS - len(remaining)
        remaining.extend([a3c_rollout_grad.remote(params, average_reward.val()) for i in range(jobs)])
        result, remaining = ray.wait(remaining)
        grads, info = ray.get(result)[0]
        average_reward.add(info["rews"])
        rwds.extend(info["real_rwds"])
        obs.extend(info["obs"])
        # print info["rews"]
            # print "DEBUG: Critic Loss - %f" % (critic.get_loss(info["obs"], info["real_rwds"]))

        # if any([(x == np.nan).any() for x in pgrad]) or any([(x == np.nan).any() for x in cgrad]):

        # if cur_itr % 50 == 0:
        #     print pgrad

        # pgrad, cgrad, info = a3c_rollout_grad(params)

        actor_critic.model_update(grads)
        cur_itr += int(info["done"])

        if cur_itr % 500 == 0:
            testbed = gym.make('CartPole-v0')
            print "%d: Avg Reward - %f" % (cur_itr, evaluate_policy(testbed, actor_critic))
            c_val = np.asarray(discounted_cumsum(rwds, GAMMA))
            c_est = actor_critic.get_value(obs).flatten()
            print "%d: Critic Loss - total: %f \t avg: %f \t most: %f" % (cur_itr, 
                sq_loss(c_val, c_est), 
                sq_loss(c_val, c_est) / len(c_val), 
                sq_loss(c_val[:-10], c_est[:-10]) / (len(c_val) - 10 + 1e-2))
            # print np.vstack([c_val, c_est]).T
            # TODO: Get average Value Fn fit
            cur_itr += 1
        if info["done"]:
            rwds = []
            obs = []

if __name__ == '__main__':
    train()
