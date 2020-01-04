import numpy as np
import math

iris = np.loadtxt('iris.txt')

#print(feature_means)
######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

def euclidean(x, Y):
    return np.sqrt(np.sum(np.power(x-Y,2),axis=1))

def split_dataset(iris):
    train_set = np.zeros(shape=(1,5))
    validation_set = np.zeros(shape=(1,5))
    test_set = np.zeros(shape=(1,5))

    for (i, ex) in enumerate(iris):
        if i%5 == 3:
            validation_set = np.append(validation_set, [ex], axis=0)
        elif i%5 == 4:
            test_set = np.append(test_set, [ex], axis=0)
        else:
            train_set = np.append(train_set, [ex], axis=0)

    train_set = train_set[1:,:]
    validation_set = validation_set[1:,:]
    test_set = test_set[1:,:]
    
    return train_set, validation_set, test_set

class basic_functions:

    def feature_means(self, iris):
        iris_feature = iris[:,:-1]
        feature_means = np.mean(iris_feature,axis=0)

        return feature_means

    def covariance_matrix(self, iris):
        iris_feature = iris[:,:-1]
        feature_covariance = np.cov(iris_feature,rowvar=False)

        return feature_covariance

    def feature_means_class_1(self, iris):
        iris_1 = np.where(iris[:,-1]==1)
        feature1_rows = iris[iris_1]
        iris_feature = feature1_rows[:,:-1]
        feature_means = np.mean(iris_feature,axis=0)

        return feature_means

    def covariance_matrix_class_1(self, iris):
        iris_1 = np.where(iris[:,-1]==1)
        feature1_rows = iris[iris_1]
        iris_feature = feature1_rows[:,:-1]
        feature_covariance = np.cov(iris_feature,rowvar=False)

        return feature_covariance

class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(np.unique(train_labels))

    def compute_predictions(self, test_data):
        #test_feature = test_data[:,:-1]
        #test_labels = test_data[:,-1]
        predicted = np.zeros(shape=(len(test_data)))

        for (i, ex) in enumerate(test_data):
            distances = euclidean(ex, self.train_inputs)
            
            #find closest neighbours
            neighbours_list = np.where(distances<self.h)
            
            if len(neighbours_list[0]) == 0:
                label_class = draw_rand_label(ex,self.n_classes)
                #predicted[i,0] = i
                predicted[i] = label_class+1
            else:
                neighbours_class = self.train_labels[neighbours_list[0]].astype(int)
                label_class = np.bincount(neighbours_class).argmax()
                #predicted[i,0] = i
                predicted[i] = label_class

        return predicted

class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels.astype(int)
        self.n_classes = len(np.unique(train_labels))

    def compute_predictions(self, test_data):
         #test_feature = test_data[:,:-1]
        #test_labels = test_data[:,-1]
        predicted = np.zeros(shape=(len(test_data)))
        d = len(self.train_inputs[0])

        for (i, ex) in enumerate(test_data):
            distances = euclidean(ex, self.train_inputs)
            neighbour_weight = np.zeros(shape=(len(distances)))
            
            for x in range(len(distances)):
                neighbour_weight[x] = (1/((2*math.pi)**(d/2))*(self.sigma**d))*(math.exp((-1/2)*((distances[x]**2)/(self.sigma**2))))
             
            n_values = np.max(self.train_labels) 
            res = np.eye(n_values)[self.train_labels-1]
            weight_matrix = np.transpose(np.multiply(np.transpose(res),neighbour_weight))
            weight_array = np.sum(weight_matrix, axis=0)
            label_class = np.argmax(weight_array) +1
            
            predicted[i] = label_class
            
        return predicted

class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        f = HardParzen(h)
        f.train(self.x_train,self.y_train)
        result = f.compute_predictions(self.x_val)
       
        div = result/self.y_val
        res = np.where(div == 1)[0]
        error_rate = 1- (len(res)/len(self.y_val))
        
        return error_rate

    def soft_parzen(self, sigma):
        f = SoftRBFParzen(sigma)
        f.train(self.x_train,self.y_train)
        result = f.compute_predictions(self.x_val)
        
        div = result/self.y_val
        res = np.where(div == 1)[0]
        error_rate = 1- (len(res)/len(self.y_val))
        
        return error_rate

def get_test_errors(iris):
    hyper_param = np.array([0.001,0.01,0.1,0.3,1.0,3.0,10.0,15.0,20.])
    train,val,test = split_dataset(iris)
    f = ErrorRate(train[:,:-1],train[:,-1],val[:,:-1],val[:,-1])
    
    h_ar = np.zeros(shape=(len(hyper_param)))
    sigma_ar = np.zeros(shape=(len(hyper_param)))
    
    for x in range(len(hyper_param)):
        hard_error = f.hard_parzen(hyper_param[x])
        soft_error = f.soft_parzen(hyper_param[x])
        
        h_ar[x] = hard_error
        sigma_ar[x] = soft_error
     
    lowest_h = hyper_param[np.argmin(h_ar)]
    lowest_sigma = hyper_param[np.argmin(sigma_ar)]
    
    f2 = ErrorRate(train[:,:-1],train[:,-1],test[:,:-1],test[:,-1])
    
    hard_Lowest_error = f2.hard_parzen(lowest_h)
    soft_lowest_error = f2.soft_parzen(lowest_sigma)
    
    return np.array([hard_Lowest_error,soft_lowest_error])

def random_projections(X, A):
    matrix_dot = np.dot(X,A)
    projection = (1/math.sqrt(2))*matrix_dot
    
    return projection
  
def test_error_random_project(loop):
    classification_results_hard_parzen = np.zeros(shape=(loop,9))
    classification_results_soft_parzen = np.zeros(shape=(loop,9))
    hyper_param = np.array([0.001,0.01,0.1,0.3,1.0,3.0,10.0,15.0,20.])
    train,val,test = split_dataset(iris)
    
    for x in range (0,loop):
        A = np.random.normal(0,1,(4,2))
        train_reduce = random_projections(train[:,:-1],A)
        test_reduce = random_projections(test[:,:-1],A)
        
        for i in range(len(hyper_param)):
            f = ErrorRate(train_reduce,train[:,-1],test_reduce,test[:,-1])
            error_hard_parzen = f.hard_parzen(hyper_param[i])
            error_soft_parzen = f.soft_parzen(hyper_param[i])
            classification_results_hard_parzen[x,i] = error_hard_parzen
            classification_results_soft_parzen[x,i] = error_soft_parzen
            
    return classification_results_hard_parzen,classification_results_soft_parzen



















