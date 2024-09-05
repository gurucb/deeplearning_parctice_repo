# URLs:

# 1. https://medium.com/@gauravnair/the-spark-your-neural-network-needs-understanding-the-significance-of-activation-functions-6b82d5f27fbf
# 2. https://medium.com/@sue_nlp/what-is-the-softmax-function-used-in-deep-learning-illustrated-in-an-easy-to-understand-way-8b937fe13d49

import numpy as np
class Utilities:
    def softmax(self,input_vector):
        total = np.sum(np.exp(input_vector))
        output = np.exp(input_vector)/total
        return output
    def sigmoid(self,input_vector):
        output = 1/(1+np.exp(-1 * input_vector))
        return output
    def relu(self,input_vector):
        pass
    def leaky_relu(self,input_vector):
        pass
    def tanh(self,input_vector):
        exp = np.exp(2*input_vector)
        output = (exp - 1)/(exp + 1)
        return output
    def argmax(self,input_vector):
        pass