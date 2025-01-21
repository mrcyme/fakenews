import numpy as np
import lineartreemod
from policy import Exp3Policy
from tools import *


class Collective():
    def __init__(self, k, policy, n_experts,  heuristic=True):
        self.policy = policy
        self.heuristic = heuristic
        self.k = k
        self.n = n_experts
        self._value_estimates = np.zeros(self.k)
        self._probabilities = np.zeros(self.k)
        self.initialize_model()
        self.t = 0

    def initialize_model(self):
        pass

    def reset(self):
        self.t = 0
        self.initialize_model()

    def get_weights(self, contexts):
        return np.ones((self.n, self.k))/self.n

    def choose(self, advice, greedy=False):
        return self.policy.choose(self, advice, greedy=greedy)

    def probabilities(self, contexts):
        self.advice = np.copy(contexts['advice'])
        assert np.shape(self.advice) == (self.n, self.k)
        if isinstance(self.policy, Exp3Policy):
            self.advice = greedy_choice(self.advice, axis=1)

        w = self.get_weights(contexts)
        self._probabilities = np.sum(w * self.advice, axis=0)

        np.testing.assert_allclose(np.sum(self._probabilities), 1, atol=1e-5)
        return self._probabilities

    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        self._value_estimates = np.sum(
            self.get_weights(contexts) * (self.advice), axis=0)

        return self._value_estimates

    def update(self, rewards, arm):
        self.observe(rewards[arm], arm)
        if self.heuristic:
            for a in range(self.k):
                if a != arm:
                    self.observe((.5*self.k-rewards[arm])/(self.k-1), a)

    def observe(self, reward, arm):
        self.t += 1


class Exp4(Collective):
    def __init__(self, k, policy, n_experts, gamma=None):
        super(Exp4, self).__init__(k, policy,
                                   n_experts, )
        self.gamma = gamma

    def initialize_model(self):
        self.e_sum = 1
        self.w = np.ones(self.n)/self.n

    def reset(self):
        super(Exp4, self).reset()

    def get_weights(self, contexts):
        w = np.repeat(np.copy(self.w)[:, np.newaxis], self.k, axis=1)
        return w

    def observe(self, reward, arm):

        assert np.allclose(np.sum(self.advice, axis=1),
                           1), "expected probability advice"
        x_t = self.advice[:, arm] * (.5-reward)

        y_t = x_t / (self.policy.pi[arm]+self.gamma)

        lr = 2*self.gamma*self.k

        y_hats = -lr * y_t
        y_hats -= np.max(y_hats)

        self.w *= np.exp(y_hats)
        self.w /= np.sum(self.w)

        self.t += 1


class OnlineRidge():
    def __init__(self, alpha, fit_intercept=False):
        self._model = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    @property
    def model(self):
        if self._model is None:
            self._model = self._init_model({})
        return self._model

    @property
    def gamma(self):
        return 1-self.weights_decay

    def _init_model(self, model):
        model['A'] = np.identity(self.context_dimension) * self.alpha
        model['A_inv'] = np.identity(self.context_dimension)/self.alpha
        model['b'] = np.zeros((self.context_dimension, 1))
        model['theta'] = np.zeros((self.context_dimension, 1))

        return model

    def fit(self, X, Y):
        self._model = None
        self.model
        self.partial_fit(X, Y)

    def partial_fit(self, X, Y):

        if self.fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.context_dimension = np.shape(X)[1]

        for x, y in zip(X, Y):
            x = x[..., None]
            self.model['A'] += x.dot(x.T)
            self.model['A_inv'] = SMInv(self.model['A_inv'], x, x, 1)
            self.model['b'] += (y) * x

        self.model['theta'] = (self.model['A_inv'].dot(self.model['b']))

    def uncertainties(self, X, sample_uncertainty=False):

        if self.fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        values = np.sqrt(
            ((X[:, :, None]*self.model['A_inv'][None, ]).sum(axis=1)*X).sum(-1))

        if sample_uncertainty:
            values = np.random.normal(np.zeros_like(
                values), values, size=values.shape)

        assert not np.isnan(values).any(), values
        return np.asarray(values)

    def predict(self, X):
        if self.fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.context_dimension = X.shape[1]

        theta = self.model['theta'][None, ]
        return (X*theta[:, :, 0]).sum(-1)


class MetaCMAB(Collective):
    def __init__(self, k, policy, n_experts, beta=None,
                 alpha=None,  fit_intercept=False,
                 residual=True, mode='UCB'):

        super().__init__(k, policy, n_experts)
        self.alpha = alpha
        self.beta = beta
        self._model = None
        self.mode = mode
        self.fit_intercept = fit_intercept
        self.residual = residual

    @property
    def model(self):
        if self._model is None:
            self.counts = {}
            self._model = OnlineRidge(
                 np.log(self.n)/self.k if self.alpha is None else self.alpha, fit_intercept=self.fit_intercept)
        return self._model

    def get_values(self, contexts, return_std=True):

        estimated_rewards = self.model.predict(contexts)
        if return_std:
            uncertainties = self.model.uncertainties(
                contexts, sample_uncertainty=(self.mode == 'TS'))
            delta=0.01 # regret guarantees hold with probability 1-delta 
            beta = 1+np.sqrt(np.log(2/delta)/2) if self.beta is None else self.beta
            return estimated_rewards, beta*uncertainties
        else:
            return estimated_rewards

    def initialize_model(self):
        self._model = None

    def value_estimates(self, contexts,):
        self.advice = np.copy(contexts['advice'])
        assert np.array(contexts['advice']).shape == (
            self.n, self.k,), "advice matrix should be of shape N (number of experts) x K (number of arms)"

        self.meta_contexts = np.array(self.advice).T
        if self.residual:
            self.meta_contexts = (np.array(
                self.advice) - np.mean(self.advice, axis=0)).T

        mu, sigma = self.get_values(self.meta_contexts)
        if self.residual:
            mu = mu + np.mean(self.advice, axis=0)

        if self.t == 0:
            return mu
        return mu + sigma/(self.t+1)

    def reset(self):
        super().reset()

    def observe(self, reward, arm):

        arm_context = self.meta_contexts[arm]
        centered_reward = reward - \
            np.mean(self.advice[:, arm]) if self.residual else reward

        self.model.partial_fit((arm_context[None]), [(centered_reward)])

        self.t += 1

    def get_weights(self):
        return self.model['theta']


class ExpertiseTree(Collective):
    def __init__(self, k, policy, n_experts,  split_features=None, fit_intercept=False, min_impurity_decrease=0, beta=None,
                 alpha=None,  residual=True,
                 min_samples_split=2, min_samples_leaf=3, max_depth=5,  criterion='rmse',
              ):

        self.min_samples_split = min_samples_split
        self.residual = residual
        self.fit_intercept = fit_intercept
        self.split_features = split_features
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        super().__init__(k, policy,
                         n_experts)
        self.alpha = alpha
        self.beta = beta

    def initialize_model(self):
        self.reward_history = []
        self.advice_history = []
        self.context_history = []
        self._fitted = False
        self.model = None

    def value_estimates(self, contexts):

        self.advice = np.array(contexts['advice']).T
        self.decision_contexts = contexts['context']
        if self.model is None:  # create ExpertiseTree
            if self.split_features is None:
                self.split_features = np.arange(
                    self.n, self.n + np.shape(self.decision_contexts)[1])
            base_regressor = OnlineRidge(alpha=np.log(self.n)/self.k 
                                         if self.alpha is None else self.alpha, fit_intercept=self.fit_intercept)

            self.model = lineartreemod.LinearTreeRegressor(base_estimator=base_regressor, max_bins=10, criterion=self.criterion,
                                                           split_features=self.split_features, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split,
                                                           linear_features=np.arange(0, self.n), min_impurity_decrease=self.min_impurity_decrease, n_jobs=1,
                                                           max_depth=self.max_depth)

        if self.t > 0 and not self._fitted:  # fit ExpertiseTree
            residue = np.mean(self.advice_history,
                              axis=1) if self.residual else np.array([0])
            X = np.hstack((np.array(self.advice_history) -
                          residue[:, None], self.context_history))
            y = np.array(self.reward_history) - residue
            self.model.fit(X, y)
            self._fitted = True

        assert self.decision_contexts.shape[0] == self.k
        assert len(self.decision_contexts.shape) > 1
        assert self.advice.shape == (self.k, self.n)

        # predict values (including exploration sigma)
        residue = np.mean(
            self.advice, axis=1) if self.residual else np.array([0])
        X = np.hstack((self.advice - residue[:, None], self.decision_contexts))

        mu, sigma = self.get_values(X)
        mu += residue
        if self.t == 0:
            return mu

        return mu + sigma/(self.t+1)

    def get_values(self, X):
        if self.t == 0:  # no tree yet, so use base estimator
            estimator = self.model.base_estimator
            pred = estimator.predict(X[:, self.model.linear_features])
            uncertainties = estimator.uncertainties(
                X[:, self.model.linear_features])

        else:  # adapts prediction from LinearTreeRegressor to also provide uncertainties
            X = self.model._validate_data(
                X,
                reset=False,
                accept_sparse=False,
                dtype='float32',
                force_all_finite=True,
                ensure_2d=True,
                allow_nd=False,
                ensure_min_features=self.model.n_features_in_
            )

            if self.model.n_targets_ > 1:
                pred = np.zeros((X.shape[0], self.model.n_targets_))
            else:
                pred = np.zeros(X.shape[0])

            uncertainties = np.zeros(X.shape[0])
            for L in self.model._leaves.values():

                mask = lineartreemod._classes._predict_branch(X, L.threshold)
                if (~mask).all():
                    continue

                pred[mask] = L.model.predict(
                    X[np.ix_(mask, self.model._linear_features)])
                uncertainties[mask] = L.model.uncertainties(
                    X[np.ix_(mask, self.model._linear_features)], sample_uncertainty=None)
                
        delta=0.01 # regret guarantees hold with probability 1-delta 
        beta = 1+np.sqrt(np.log(2/delta)/2) if self.beta is None else self.beta
        return pred, beta*uncertainties

    def observe(self, reward, arm):
        self.reward_history.append(reward)
        self.advice_history.append(self.advice[arm])
        self.context_history.append(self.decision_contexts[arm])
        self._fitted = False
        self.t += 1
