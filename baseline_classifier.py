#!/usr/bin/env/python

import sys
from math import log

########################################################################
#                   Step 2 - Construct Bayesian Network                #
########################################################################

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
