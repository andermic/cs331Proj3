#!/usr/bin/env python

"""
Written by Michael Anderson
For CS331 - Introduction to Artificial Intelligence
Spring 2010
"""

import os
import sys
from math import log

# If the user does not provide the correct number of arguments, output a
#  usage message.
if len(sys.argv) != 5:
    print 'USAGE: classifier.py <isBaseline> <stoplist file>',
    print '<training data file> <testing data file>'
    exit()

isBaseline = sys.argv[1]
training_file_name = sys.argv[3]
testing_file_name = sys.argv[4]


########################################################################
#                   Step 2 - Construct Bayesian Network                #
########################################################################

# Get training data from file. First get the vocab list
training_file = open(training_file_name, 'r')
vocab_list = training_file.readline()
vocab_list = vocab_list.split(',')
vocab_list = vocab_list[:-1]

# Then count the occurrences of each word within a given category. Also
#  count the total number of records in each category.
counts = {}
record_nums = {}
while True:
    line = training_file.readline()
    if line == '':  # Reached end of file, so break
        break

    # Make the feature vector string a list, grab its class label, and
    #  give it its own dictionary if it does not already exist.
    line = line.split(',')
    class_label = line[-1].strip()
    if class_label not in counts:
        counts[class_label] = {}
        record_nums[class_label] = 0

    # Keep track of the total number of records found for each category
    record_nums[class_label] += 1

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

total_records = 0
categories = record_nums.keys()
for category in categories:
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

# Get testing data from file. First get the vocab list
testing_file = open(testing_file_name, 'r')
vocab_list = testing_file.readline()
vocab_list = vocab_list.split(',')
vocab_list = vocab_list[:-1]

correct_nums = {}
for category in categories:
    correct_nums[category] = 0
# For each file, predict its category using the training data. Keep
#  track of the total number of correct guesses for each category.
file_count = 0
while True:
    file_count += 1
    print 'Classifying file ' + str(file_count)
    
    line = testing_file.readline()
    if line == '':  # Reached end of file, so break
        break

    line = line.split(',')
    best_category = None
    best_prob = None

    for category in categories:
        cur_prob = 0
        for word_index in range(len(vocab_list)):
            if vocab_list[word_index] in probs[category]:
                if line[word_index] == '1':
                    cur_prob += log(probs[category][vocab_list[word_index]])
                else:
                    cur_prob += log(1-probs[category][vocab_list[word_index]])
        if not best_category or cur_prob > best_prob:
            best_category = category
            best_prob = cur_prob

    actual_category = line[-1].strip()
    if best_category == actual_category:
        correct_nums[actual_category] += 1

print
print 'RESULTS:'
for category in categories:
    print '%s - %.2f%%' % (category,
     100 * float(correct_nums[category]) / float(record_nums[category]))
