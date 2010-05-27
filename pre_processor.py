#!/usr/bin/env python

"""
Written by Michael Anderson
For CS311 - Introduction to Artificial Intelligence
Spring 2010
"""

import os
from zipfile import ZipFile
from math import log

# Takes a word and a list of words that are sorted alphabetically.
#  Performs a recursive binary search and returns True if the word is in
#  the sorted list, False if it is not.
def binary_search(word, list):
    mid = len(list) / 2
    if word == list[mid]:
        return True
    if len(list) == 1:
        return False
    elif word < list[mid]:
        return binary_search(word, list[:mid])
    elif word > list[mid]:
        return binary_search(word, list[mid:])


########################################################################
#                    Step 1 - Gather Training Data                     #
########################################################################

# Decompress the training zip file
TRAINING_ZIP_FILE = 'training_dataset.zip'
destination = TRAINING_ZIP_FILE[:TRAINING_ZIP_FILE.rfind('.')]
training_file = ZipFile(TRAINING_ZIP_FILE)
training_file.extractall(destination)
training_file.close()

# Load the stoplist into a python list
STOPLIST_FILE_NAME = 'stoplist.txt'
stoplist_obj = open(STOPLIST_FILE_NAME, 'r')
stoplist = stoplist_obj.read().split('\n')
stoplist_obj.close()

# Code from here forward assumes that the zip file contained a single
#  directory. This directory should contain subdirectories for each 
#  category. Each subdirectory should contain files associated with that
#  category.
training_root = os.path.join(destination, os.listdir(destination)[0])

categories = os.listdir(training_root)
feature_lists = []
vocab_list = set([])

# Walk through the files in the training directory with the help of some
#  python voodoo
directory_tree = os.walk(training_root)
for dir_name, subdir_names, file_names in directory_tree:
    for file_name in file_names:
        # Create a new feature list for this file
        feature_lists.append({})
        cur_list = feature_lists[len(feature_lists) - 1]
        
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
                # Check that the token is valid and not in the stoplist
                if token and not binary_search(token, stoplist):
                    # Add the token to the vocab and feature list
                    vocab_list.add(token)
                    cur_list[token] = '1'
                token = ''  # Get ready for the next token

        # Add the ClassLabel, the parent directory of this file
        cur_list['ClassLabel'] = dir_name[dir_name.rfind('/')+1:]

OUTPUT_FILE_NAME = 'training.txt'
output_file = open(OUTPUT_FILE_NAME, 'w')

'''
# Write the sorted vocabulary out to a file
vocab_list = sorted(list(vocab_list))
for word in vocab_list:
    output_file.write(word + ',')
output_file.write('ClassLabel\n')

counts = {}
for category in categories:
    counts[category] = {}
# Write the feature lists out to a file. Since we have to iterate
#  through everything here anyway, which is computationally expensive,
#  might as well count occurrences of each word in each category.
for feature_list in feature_lists:
    count_dict = counts[feature_list['ClassLabel']]
    for word in vocab_list:
        if word in feature_list:
            output_file.write(feature_list[word] + ',')
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
        else:
            output_file.write('0,')
    output_file.write(feature_list['ClassLabel'] + '\n')
output_file.close()
'''


########################################################################
#                   Step 2 - Construct Bayesian Network                #
########################################################################

# Get the total number of records, and the number per category
record_nums = {}
total_records = 0
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
         (float(count + 1) / float(record_nums[category] + 2)))


########################################################################
#                       Step 3 - Test Network                          #
########################################################################

# Decompress the testing zip file
TESTING_ZIP_FILE = 'training_dataset.zip'
destination = TESTING_ZIP_FILE[:TESTING_ZIP_FILE.rfind('.')]
testing_file = ZipFile(TESTING_ZIP_FILE)
testing_file.extractall(destination)
testing_file.close()

testing_root = os.path.join(destination, os.listdir(destination)[0])

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
                # Check that the token is valid, in the vocabulary,
                #  and not in the stoplist
                if token and not binary_search(token, stoplist) and
                 binary_search(token, vocab_list):
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
        
