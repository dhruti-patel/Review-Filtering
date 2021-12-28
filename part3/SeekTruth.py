# SeekTruth.py : Classify text objects into two categories
#
# Dhruti Patel - dsp3
#
# Based on skeleton code by D. Crandall, October 2021
#

import sys

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#
def stopWord(word):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "for", "to", "had","in"]
    if word in stop_words:
        return True
    return False

def classifier(train_data, test_data):
    truth_dict = {}
    deceptive_dict={}
    total_dict={}
    t_count=0
    d_count=0
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # Collecting information from trained data
    for i in range(len(train_data["objects"])):
        truth = False
        if train_data["labels"][i] == "truthful":
            truth = True
            t_count += 1
        else:
            d_count += 1
        for word in train_data["objects"][i].lower().split(' '):
            if not stopWord(word):
                for letter in word:
                    if letter in punc:
                        word = word.replace(letter, "")
                if word not in total_dict:
                    total_dict[word] = 0
                total_dict[word] += 1

                if truth:
                    if word not in truth_dict:
                        truth_dict[word] = 0
                    truth_dict[word] += 1
                else:
                    if word not in deceptive_dict:
                        deceptive_dict[word] = 0
                    deceptive_dict[word] += 1
    prior_truthful = t_count/len(train_data["objects"])
    prior_deceptive = d_count/len(train_data["objects"])

    # Removing words with lower frequency (tried playing around with the number such that it does not give divide by zero error and also gets the highest accuracy)
    for word in total_dict:
        if total_dict[word]<25:
            if word in truth_dict:
                truth_dict.pop(word)
            if word in deceptive_dict:
                deceptive_dict.pop(word)
    
    # Calculating Odds Ratio for each object in test data and assigning label accordingly
    test_labels=[]
    alpha = 1.5
    for j in range(len(test_data["objects"])):
        likelihood_t=prior_truthful
        likelihood_d=prior_deceptive
        unique_words= set()
        for word in test_data["objects"][j].lower().split(' '):
            for letter in word:
                if letter in punc:
                    word = word.replace(letter, "")
            unique_words.add(word)
        for u_word in unique_words:
            if u_word in truth_dict and u_word in deceptive_dict:
                likelihood_t *= truth_dict[u_word]/len(truth_dict)
                likelihood_d *= deceptive_dict[u_word]/len(deceptive_dict)
                # Tried laplace smoothing, highest accuarcy i could get is 77% using laplace smoothing. I am getting higher accuracy without using laplace smoothing 
            #     likelihood_t *= (truth_dict[u_word]+alpha)/len(truth_dict)+alpha*2
            #     likelihood_d *= (deceptive_dict[u_word]+alpha)/len(deceptive_dict)+alpha*2
            # else:
            #     likelihood_t *= alpha/len(truth_dict)+alpha*2
            #     likelihood_d *= alpha/len(deceptive_dict)+alpha*2
        if likelihood_t/likelihood_d > 1.0:
            test_labels.append("truthful")
        else:
            test_labels.append("deceptive")

    return test_labels


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results= classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
