import numpy as np
import re, argparse
import nltk
from nltk.corpus import stopwords, words 
from nltk.tokenize import WordPunctTokenizer
from scipy.sparse import csr_matrix,save_npz, hstack
from nltk.stem import WordNetLemmatizer
#from nltk.stem.porter import PorterStemmer

# Used to split the data into training and testing. The ranges argument will create folds from 
# 0-ranges, ranges - ranges+5000, etc. 
def split_data(features, labels, ranges):
    sample = np.arange(0,len(labels)).astype(int)
    exclude = np.arange(ranges,ranges+5000)
    include = np.setdiff1d(sample,exclude)

    # place seed to our random shuffling is consistent each run
    np.random.seed(123)
    np.random.shuffle(sample)
    #split = int(len(labels) * split)
    
    x_train = features[include]
    x_test = features[exclude]
    y_train = labels[include]
    y_test = labels[exclude]
    
    return x_train, x_test, y_train, y_test

#http://www.insightsbot.com/blog/R8fu5/bag-of-words-algorithm-in-python-introduction
#Function to clean the text stop words, \n and tokenize each entry into an array
def create_list_of_words(sentence):
	# Use the NLTK stopword list
    stopword_list = stopwords.words("english")
    #porter_stemmer  = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    
    #remove all \n
    remove_n = [x.replace('\n', '') for x in sentence]
    
    #extract only the words
    keep_letters = [re.sub('[^A-Za-z]+', " ",  x) for x in remove_n]
    
    # tokenize words
    tokens = WordPunctTokenizer().tokenize_sents(keep_letters)
    
    cleaned_text = []
    #remove all stop words and lemmatization
    for entry in tokens:
        # ([w.lower() for w in entry if w not in stopword_list and w in words.words()])
        remove_stopwords = [w.lower() for w in entry if w.lower() not in stopword_list]
        lem = [wordnet_lemmatizer.lemmatize(lm) for lm in remove_stopwords]
        cleaned_text.append(lem)
        
    return np.array(cleaned_text)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
# Create a bag of words in matrix form. Also return a dict of the vocabulary 
# Followed the implementation in the Scipy docs above
def bag_of_words(docs):
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for d in docs:
        for term in d:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)

        indptr.append(len(indices))

    mat = csr_matrix((data, indices, indptr), dtype=int)
    
    return mat, vocabulary

# Reduce the size of the vocabulary by only returning the top N words. Parameter controlled
# by the word_threshold argument
def reduced_size(matrix, vocabulary, word_threshold):
	#Sum the occureence of each word in our vocab accross the entries in the training data 
    sums = np.array(matrix.sum(axis=0))
    index_range = np.arange(0,sums.shape[1])
    n_sum = np.vstack((index_range,sums))
    
    #Sort and get the top N words and their index in the vocabulary
    sorted_stack_mat = n_sum[:,n_sum[1,:].argsort()[::-1]]
    final_list = sorted_stack_mat[:,:]
    final_indexes = final_list[0,:]
    
    #Create a new mat only holding the words above the threshold
    new_mat = matrix[:,final_indexes]
    
    # swap the KV to get the index as key and the word as value
    res = dict((v,k) for k,v in vocabulary.items())
    
    # Create a dict with the adjusted word index from the new matrix created
    new_vocab = {}
    for i,index in enumerate(final_indexes):
        new_vocab[res[index]] = i
        
    return new_mat, new_vocab

# Function which calculates the prior and probability of our training data
def create_prob(features, labels, unique_labels, vocab):
	prob = np.zeros(shape=(len(unique_labels),features.shape[1])) 
	priors = np.zeros(len(unique_labels))

	# Loop through each class in our dataset
	for i,lb in enumerate(unique_labels):
		#Get the index of rows for said class 
	    label_index = list(np.where(labels == lb)[0])

	    #Get the sum of each word in our vocab for said class
	    sum_of_words = features[label_index,:].sum(axis=0)
	    #Get the len of the vocab
	    len_of_vocab = len(vocab)

	    #Total number of words in said class
	    words_n = sum_of_words.sum()

	    #Calculate the log prob as per Naive Bayes with smoothing
	    prob[i,:] = np.log((sum_of_words+0.2)/(words_n+0.2*(len_of_vocab)))

	    #Calculate log priors of said class
	    priors[i] = np.log(len(label_index)/len(labels))
	    
	return prob, priors

# Classify the unseen data
def classify(test_features, prob, priors, unique_labels, vocab):
    # Create empty list of the size of our test sample 
    pr = ['']*len(test_features)

    # For each entry in our test sample, return the class with the highet probability 
    for i,entry in enumerate(test_features):
    	#Using the vocab, get the index of the words for said entry
        word_index = [vocab.get(key) for key in entry]

        # drops Nones (words not part of our training data)
        word_ind = [x for x in word_index if x is not None]

        # Compute the probability by multiplying conditional*priors
        probs= np.sum(prob[:,word_ind], axis=1)+priors
        #Return class with highest prob
        pred = np.argmax(probs)
        
        pr[i] = unique_labels[pred]
        #print(pr[i])

    return pr

# Compute classification rate
def classification_rate(pr,y_test):
    return np.mean(pr == y_test)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training and testing file')
	parser.add_argument("-train")
	parser.add_argument('-test')
	args = parser.parse_args()

	if args.train is None or args.test is None:
		print('Using default hardcoded training and testing test.')
		data = np.load('data_train.pkl', allow_pickle=True)
		tsentences = np.load('data_test.pkl', allow_pickle=True)
	else:
		print('Using '+str(args.train)+' as the training set and '+str(args.test)+' as the testing set.')
		data = np.load(args.train, allow_pickle=True)
		tsentences = np.load(args.test, allow_pickle=True)

	sentences = data[0]
	lbs = data[1]

	labels = np.array(lbs)
	unique_labels = np.unique(labels)

	encoded_labels = np.zeros(len(lbs))
	for i,lb in enumerate(unique_labels):
	    encoded_labels[np.where(labels==lb)] = i

	encoded_labels = encoded_labels.astype(int)
	unique_encoded_labels = np.unique(encoded_labels)

	# for stopword
	nltk.download('stopwords')

	# for word_tokenize
	nltk.download('punkt')

	# for english dictionary
	#nltk.download('words')
	
	# for lem
	nltk.download('wordnet')


	cleaned_data = create_list_of_words(sentences)
	tcleaned_data = create_list_of_words(tsentences)

	obj,vocab = bag_of_words(cleaned_data)

	new_mat, new_vocab = reduced_size(obj, vocab, 50000)
	prob, priors = create_prob(new_mat, encoded_labels, unique_encoded_labels, new_vocab)
	prediction = classify(tcleaned_data, prob, priors, unique_encoded_labels, new_vocab)

	predictions = ['']*(len(prediction)+1)
	predictions[0] = 'Id,Category'
	for i,p in enumerate(prediction):
	    predictions[i+1] = str(i)+','+str(unique_labels[p])

	np.savetxt("sample_submission.csv", predictions, delimiter=",",fmt='%s')
	