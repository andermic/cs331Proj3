#!/usr/bin/env python

import os
import sys
from math import log

########################################################################
#                   Step 2 - Construct Bayesian Network                #
########################################################################

# Get training data from file. First get the vocab list
TRAINING_FILE = 'training.txt'
training_file = open('training.txt', 'r')
vocab_list = training_file.readline()
vocab_list = vocab_list.split(',')
vocab_list = vocab_list[:-1]

# Then count the occurrences of each word within a given category. Also
#  count the total number of records in each category.
counts = {}
while True:
    line = training_file.readline()
    if line == '':  # Reached end of file, so break
        break

    # Make the feature vector string a list, grab its class label, and
    #  give it its own dictionary if it does not already exist.
    line = line.split(',')
    class_label = line[-1]
    if class_label not in counts:
        counts[class_label] = {}

    # Iterate through the feature list
    for word_index in range(len(vocab_list)):
        if line[word_index] == '1':
            # Increment the word count, or create it if it does not yet
            #  exist
            if vocab_list[word_index] in counts[class_label]:
                counts[class_label][vocab_list[word_index]] += 1
            else:
                counts[class_label][vocab_list[word_index]] = 1

training_file.close()

record_nums = {}
total_records = 0
categories = counts.keys()
for category in categories:
    record_nums[category] = \
     len(os.listdir(os.path.join(training_root, category)))
    total_records += record_nums[category]
probs = {}
cprobs = {}
# Calculate the expected probability that a random record will belong to
#  a category. Calculate the expected probability that a word will occur
#  in a record of a given category. 
for category in categories:
    probs[category] = {}
    cprobs[category] = float(record_nums[category]) / float(total_records)
    for word in vocab_list:
        count = 0
        if word in counts[category]:
            count = counts[category][word]

        # Use Dirichlet Priors trick in calculation.
        probs[category][word] = \
         (float(count + 1) / float(record_nums[category] + 2))


########################################################################
#                       Step 3 - Test Network                          #
########################################################################

'''
scores = {}
# Walk through the files in the testing directory with the help of some
#  python voodoo
directory_tree = os.walk(testing_root)
for dir_name, subdir_names, file_names in directory_tree:
    if dir_name != testing_root:
        scores[category] = [0, len(os.listdir(dir_name))]
    for file_name in file_names:
        # Create a new feature list for this file
        cur_list = {}
        
        # Get the contents of the given file. Make all letters
        #  lowercase.
        stream = open( os.path.join(dir_name, file_name), 'r' )
        file = stream.read()
        file = file.lower()

        # Tokenize the file contents
        token = ''
        for char in file:
            # Check if the character is a member of the alphabet. If so,
            #  append it to the current token.
            if ord(char) >= ord('a') and ord(char) <= ord('z'):
                token += char
            else:
                # Check that the token is valid and in the vocabulary
                if token and binary_search(token, vocab_list):
                    cur_list[token] = '1'
                token = ''  # Get ready for the next token

        # FINISH AND DOUBLECHECK ME!
        # I PREDICT THE CATEGORY OF THE GIVEN FILE!
        best_cat = None
        best_score = None
        for category in categories:
            cur_score = 0
            for word in vocab_list:
                if word in cur_list:
                    total_probs['category'] += log(probs[category][word])
                else:
                    total_prob['category'] += log(1 - probs[category][word])
            if not best_cat or best_score > cur_score:
                best_cat = category
                best_score = cur_score
'''
