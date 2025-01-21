
import numpy as np

from tools import greedy_choice, softmax



class Policy(object):
    def __init__(self, b=1):
        self.b = b
        self.key = 'value'

    def __str__(self):
        return 'generic policy'

    def probabilities(self, agent, contexts):
        a = agent.value_estimates(contexts)
        self.pi = softmax(a*self.b)
        return self.pi
        

    def choose(self, agent, contexts, greedy=False):
        

        self.pi = self.probabilities(agent, contexts)
        np.testing.assert_allclose(np.sum(self.pi),1,atol=1e-5,err_msg=str(agent)+" "+str(np.sum(self.pi))+" "+str(self.pi))
        
        if greedy:
            self.pi = greedy_choice(self.pi)
            
        np.testing.assert_allclose(np.sum(self.pi),1,atol=1e-5,err_msg=str(agent)+" "+str(np.sum(self.pi))+" "+str(self.pi))
        action = np.searchsorted(np.cumsum(self.pi), np.random.rand(1))[0]
        
        return action
        

class EpsilonGreedyPolicy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.key = 'value'

    def __str__(self):
        return 'eps'.format(self.epsilon)

    def probabilities(self, agent, contexts):
        self.v = agent.value_estimates(contexts)
        
        self.pi = (1-self.epsilon)*greedy_choice(self.v)     +self.epsilon/agent.k  
    
        return self.pi


    @staticmethod
    def choice(values):
        pi = greedy_choice(values)       
        np.testing.assert_allclose(np.sum(pi),1,atol=1e-5,err_msg=str(values)+" "+str(np.sum(pi))+" "+str(pi))
        action = np.searchsorted(np.cumsum(pi), np.random.rand(1))[0]

        return action


class GreedyPolicy(EpsilonGreedyPolicy):

    def __init__(self):
        super().__init__(0)

    def __str__(self):
        return 'greedy'


class Exp3Policy(Policy):
    def __init__(self, eps=0):
        self.eps = eps
        self.key = 'probability'

    def __str__(self):
        return 'E3P'

    def probabilities(self, agent, contexts):
        self.pi = agent.probabilities(contexts)
        np.testing.assert_allclose(np.sum(self.pi),1,atol=1e-5,err_msg=str(agent)+" "+str(np.sum(self.pi))+" "+str(self.pi))
        
        self.pi = self.pi * (1 - self.eps) + self.eps / len(self.pi) 
        return self.pi
