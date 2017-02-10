
import gym
import numpy as np
import tensorflow as tf
from misc import discounted_cumsum, policy_rollout
import time
from tfmodel import TFModel
# from critic import Critic
import ray
from collections import defaultdict

class ActorCritic(TFModel):

    def __init__(self, hparams):

        start = time.time()

        # build the graph
        self.g = tf.Graph()
        with self.g.as_default():
            self.setup_graph(hparams)
            self.setup_loss_fns(hparams)

            # we can do some sharing here
            self.optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'],
                                                        decay=0.99,
                                                        epsilon=0.1,
                                                        momentum=7e-4
                                                        )
            self._train = self.optimizer.minimize(self.loss)

            self.grads_vars = self.optimizer.compute_gradients(self.loss)
            self._apply_gradients = self.optimizer.apply_gradients(self.grads_vars)
            self.initializer = tf.global_variables_initializer()
            print "AC Model took %f seconds to load..." % (time.time() - start)
        
        self.start()

    def setup_graph(self, hparams):
        self._input = tf.placeholder(tf.float32,
                shape=[None, hparams['input_size']])

        hidden1 = tf.contrib.layers.fully_connected(
                inputs=self._input,
                num_outputs=hparams['hidden_size'],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
        hidden2 = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=hparams['hidden_size'],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.contrib.layers.fully_connected(
                inputs=hidden2,
                num_outputs=hparams['num_actions'],
                activation_fn=None)

        self._value = tf.contrib.layers.fully_connected(
                inputs=hidden2,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())

        # op to sample an action - multinomial takes unnormalized log probs
        # self.logits = tf.Print(self.logits, [self.logits], "self.Logits - ")
        self._sample = tf.reshape(tf.multinomial(self.logits, 1), [])



    def setup_loss_fns(self, hparams):
        # Policy Gradient Loss
        self._acts = tf.placeholder(tf.int32)
        self._advantages = tf.placeholder(tf.float32)
        log_prob = tf.log(tf.nn.softmax(self.logits)) #this is negative
        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)#this is negative
        entropy = -tf.reduce_mean(tf.mul(tf.nn.softmax(self.logits), log_prob))
        policy_loss = -(tf.reduce_mean(tf.mul(act_prob, self._advantages)) 
                                        + hparams['entropy_wt'] * entropy)


        # Value Function Loss
        self._discounted_rwds = tf.placeholder(tf.float32)
        v = tf.reshape(self._value, [-1]) - self._discounted_rwds
        self.valuefn_loss = tf.reduce_sum(tf.square(v))

        # update
        self.loss = policy_loss + 0.5 * self.valuefn_loss # according to Mnih

    def start(self):
        self._s = tf.Session(graph=self.g)
        with self.g.as_default():
            self.variables = ray.experimental.TensorFlowVariables(self.loss, self._s)   
        self._s.run(self.initializer)
        # self.variables = ray.experimental.TensorFlowVariables(self.loss, self._s, prefix=True)
        self.g.finalize()
        self.debug_info = defaultdict(list)


    def act(self, observation):
        # get one action, by sampling
        return self._s.run(self._sample, feed_dict={self._input: [observation]})


    def get_vf_loss(self, obs, discounted_rwds):
        batch_feed = { self._input: obs, \
                self._discounted_rwds: discounted_rwds }
        return self._s.run(self.valuefn_loss, feed_dict=batch_feed)

    def get_value(self, obs):
        batch_feed = { self._input: obs }
        return self._s.run(self._value, feed_dict=batch_feed)


    def compute_gradients(self, obs, acts, advantages, discounted_rwds, debug=False):
        batch_feed = { self._input: obs, \
                self._acts: acts, \
                self._advantages: advantages,
                self._discounted_rwds: discounted_rwds  }
        if debug:
            import ipdb; ipdb.set_trace()  # breakpoint 17a2637c //

        return self._s.run([g[0] for g in self.grads_vars], feed_dict=batch_feed)

    def model_update(self, grads):
        feed_dict = {self.grads_vars[i][0]: grads[i] 
                            for i in range(len(grads))}
        self._s.run(self._apply_gradients, feed_dict=feed_dict)

    def get_weights(self, debug=False):
        weights = self.variables.get_weights()
        if debug:
            self.log_weights(weights)
        return weights

    def set_weights(self, weights, debug=False):
        if debug:
            self.log_weights(weights)
        self.variables.set_weights(weights)

    def log_weights(self, weights):
        if len(self.debug_info['weights']) > 10:
            self.debug_info['weights'].pop(0)
        self.debug_info['weights'].append(weights)
