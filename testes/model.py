from abc import ABC, abstractmethod

### Auxiliar Class
class AbstractModel(ABC):
    '''
    Abstract class representing a Neural Network Model.
    '''
    @abstractmethod
    def get_layers(self):
        '''
        Returns a list where each item corresponds to the weights (parameters) of a layer.
        The order of the elements should be the same for every call.
        For example, this function could call model.parameters() in torch or
        model.get_weights() in tensorflow.
        '''

    @abstractmethod
    def set_layers(self, new_layers):
        '''
        Replace current model weights with the new_weights layer by layer.
        '''

    @abstractmethod
    def train(self):
        '''
        Train the model.
        '''
