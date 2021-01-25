
from keras.layers.core import Layer  
from keras import initializers, regularizers, constraints  
from keras import backend as K

class ELMo(Layer):
    def __init__(self,
                 kernel_regularizer=None, 
                 kernel_constraint=None, 
                 **kwargs):
        self.supports_masking = True
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        super(ELMo, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight((input_shape[-1], 1),
                                 initializer='zeros',
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.gamma = self.add_weight((),
                                 initializer='ones',
                                 name='{}_b'.format(self.name))

        self.built = True

    def compute_mask(self, x, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
#        Edot = K.dot(x, self.kernel)
#        Edot = K.squeeze(Edot, -1)
        #s = K.relu(self.kernel)
        print(self.gamma)
        print(self.kernel)
        #s = K.softmax(self.kernel+1.0/3,dim=0)
        #s = K.softmax(self.kernel+1.0/3,axis=0)
        s = K.softmax(self.kernel+1.0/3)
        print(s)
        weighted_input =K.dot(x, s)
        print(weighted_input)
        ELMo = K.squeeze(weighted_input, -1)
        print(ELMo)
        ELMo = ELMo*self.gamma
        print(ELMo)
        return ELMo

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2])
