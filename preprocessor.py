#!/usr/bin/env python

"""
Written by Michael Anderson & Sean Moore
For CS331 - Introduction to Artificial Intelligence
Spring 2010

Formats a group of zipped files for use in a Bayesian classifier
"""

import os
import sys
from zipfile import ZipFile

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

# If the user does not provide the correct number of arguments, output a
#  usage message.
if len(sys.argv) != 3:
    print 'USAGE: preprocessor.py <zipped directory> <advanced1> <advanced2>'
    exit()

zipped_dir = sys.argv[1]

# To throw out words that occur less than k times, give the word "true"
#  (without quotes) as the advanced1 command line argument.
advanced1 = sys.argv[2]

# To add the k most frequently occurring bigrams and trigrams to the
#  vocabulary list, give the word "true" (without quotes) as the
#  advanced2 argument.
'''advanced2 = sys.argv[3]'''


########################################################################
#                    Step 1 - Gather Training Data                     #
########################################################################

# Decompress the training zip file
destination = zipped_dir[:zipped_dir.rfind('.')]
training_file = ZipFile(zipped_dir)
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
vocab_dict = {}

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
            # Check if the character is alphanumerical. If so, append it
            #  to the current token.
            if ( ord(char) >= ord('a') and ord(char) <= ord('z') ) or \
               ( ord(char) >= ord('0') and ord(char) <= ord('9') ):
                token += char
            else:
                # Check that the token is valid and not in the stoplist
                if token and not binary_search(token, stoplist):
                    # Add the token to the vocab dict and feature list
                    if token not in vocab_dict:
                        vocab_dict[token] = 1
                    else:
                        vocab_dict[token] += 1
                    cur_list[token] = '1'
                token = ''  # Get ready for the next token

        # Add the ClassLabel, the parent directory of this file
        cur_list['ClassLabel'] = dir_name[dir_name.rfind('/')+1:]

# If the user chooses to run the advanced processor, then eliminate all
#  words in the vocab dict that occur less than K_VALUE times.
if advanced1 == 'true':
    K_ADVANCED1 = 5
    for key in vocab_dict.keys():
        if vocab_dict[key] < K_ADVANCED1:
            del(vocab_dict[key])

output_file_name = zipped_dir[:zipped_dir.find('_')] + '.txt'
output_file = open(output_file_name, 'w')

# Write the sorted vocabulary out to a file
vocab_list = sorted(vocab_dict.keys())
for word in vocab_list:
    output_file.write(word + ',')
output_file.write('ClassLabel\n')

# Write the feature lists out to a file
for feature_list in feature_lists:
    for word in vocab_list:
        if word in feature_list:
            output_file.write(feature_list[word] + ',')
        else:
            output_file.write('0,')
    output_file.write(feature_list['ClassLabel'] + '\n')
output_file.close()
