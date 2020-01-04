import pickle
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    #DONE
    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            lower = -1/np.sqrt(all_dims[layer_n-1])
            upper = 1/np.sqrt(all_dims[layer_n-1])
            self.weights[f"W{layer_n}"] = np.random.uniform(lower,upper,size=(all_dims[layer_n-1],all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
    #DONE
    def relu(self, x, grad=False):
        x = np.array(x)
        if grad:
            x[x<0] = 0
            x[x>0] = 1
            
            return x
        
        return np.maximum(0,x)
    #DONE
    def sigmoid(self, x, grad=False):
        if grad:
            return (np.exp(-x))/((1 + np.exp(-x))**2)

        return 1/(1 + np.exp(-x))
    #DONE
    def tanh(self, x, grad=False):
        if grad:
            return (4*np.exp(2*x))/((np.exp(2*x)+1)**2)
        
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    #DONE
    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x,grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x,grad)
        elif self.activation_str == "tanh":
            return self.tanh(x,grad)
        else:
            raise Exception("invalid")
    #DONE
    # https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        if x.ndim == 1:
            x = x - np.max(x)
            num = np.exp(x)
            return num/np.sum(num)
        else:
            x = x - np.max(x, axis=1)[:, np.newaxis]
            num = np.exp(x)
            return num/np.sum(num, axis=1, keepdims=True)
    #DONE
    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        last_layer = self.n_hidden + 1

        for layer_n in range(1, last_layer):
            cache[f"A{layer_n}"] = np.dot(cache[f"Z{layer_n-1}"], self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"] 
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])

        cache[f"A{last_layer}"] = np.dot(cache[f"Z{last_layer-1}"], self.weights[f"W{last_layer}"]) + self.weights[f"b{last_layer}"] 
        cache[f"Z{last_layer}"] = self.softmax(cache[f"A{last_layer}"])
            
        return cache
    #DONE
    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        batch = output.shape[0]
        grads = {}
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        cross_entropy_grad = output - labels
        
        grads[f"dA{self.n_hidden + 1}"] = cross_entropy_grad #* self.activation(cache[f"A{self.n_hidden + 1}"], grad=True)
        grads[f"dW{self.n_hidden + 1}"] = 1/batch * np.dot(cache[f"Z{self.n_hidden}"].T, grads[f"dA{self.n_hidden + 1}"])
        grads[f"db{self.n_hidden + 1}"] = 1/batch * np.sum(grads[f"dA{self.n_hidden + 1}"], axis=0, keepdims=True)
 
        for layer_n in range(self.n_hidden , 0, -1):
            grads[f"dZ{layer_n}"] = np.dot(grads[f"dA{layer_n+1}"], self.weights[f"W{layer_n+1}"].T)            
            grads[f"dA{layer_n}"] = grads[f"dZ{layer_n}"] * self.activation(cache[f"A{layer_n}"], grad=True)
            grads[f"dW{layer_n}"] = 1/batch * np.dot(cache[f"Z{layer_n-1}"].T, grads[f"dA{layer_n}"])
            grads[f"db{layer_n}"] = 1/batch * np.sum(grads[f"dA{layer_n}"], axis=0, keepdims=True)

        
        return grads
    #DONE
    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] -= self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= self.lr * grads[f"db{layer}"]
    #DONE
    #https://stackoverflow.com/a/29831596/9980065
    def one_hot(self, y):
        oh = np.zeros((len(y), self.n_classes))
        oh[np.arange(len(y)), y] = 1
        return oh
    #DONE
    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        
        losses = -np.log(np.sum(labels*prediction,axis=1))
        loss = np.sum(losses)/len(losses)
        
        return loss
    #DONE
    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions
    #DONE
    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                fw = self.forward(minibatchX)
                bw = self.backward(fw, minibatchY)
                self.update(bw)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs
    #DONE
    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy

   