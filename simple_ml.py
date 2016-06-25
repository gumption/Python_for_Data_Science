from __future__ import print_function, division

''' Utility functions to implement some simple Machine Learning tasks
'''

__author__ = 'Joe McCarthy'
__version__ = '1.0.3'
__date__ = '2014-04-04'
__maintainer__ = 'Joe McCarthy'
__email__ = 'joe@interrelativity.com'
__status__ = 'Development'


import math
import operator

from collections import defaultdict, Counter


def load_instances(filename, filter_missing_values=False, missing_value='?'):
    '''Returns a list of instances stored in a file.
    
    filename is expected to have a series of comma-separated attribute values per line, e.g.,
        p,k,f,n,f,n,f,c,n,w,e,?,k,y,w,n,p,w,o,e,w,v,d'''
    instances = []
    with open(filename, 'r') as f:
        for line in f:
            new_instance = line.strip().split(',')
            if not filter_missing_values or missing_value not in new_instance:
                instances.append(new_instance)
    return instances


def save_instances(filename, instances):
    '''Saves a list of instances to a file.
    
    instances are saved to filename one per line, 
    where each instance is a list of attribute value strings.'''
    with open(filename, 'w') as f:
        for instance in instances:
            f.write(','.join(instance) + '\n')


def load_attribute_names(filename, separator=':'):
    '''Returns a list of attribute names in a file.
    
    filename is expected to be a file with attribute names. one attribute per line.
    
    filename might also have a list of possible attribute values, in which case it is assumed
    that the attribute name is separated from the possible values by separator.'''
    with open(filename, 'r') as f:
        attribute_names = [line.strip().split(separator)[0] for line in f]
    return attribute_names


def load_attribute_values(attribute_filename):
    '''Returns a list of attribute values in filename.
    
    The attribute values are represented as dictionaries, 
    wherein the keys are abbreviations and the values are descriptions.
    
    filename is expected to have one attribute name and set of values per line, 
    with the following format:
        name: value_description=value_abbreviation[,value_description=value_abbreviation]*
    for example
        cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
    The attribute value description dictionary created from this line would be the following:
        {'c': 'conical', 'b': 'bell', 'f': 'flat', 'k': 'knobbed', 's': 'sunken', 'x': 'convex'}'''
    attribute_values = []
    with open(attribute_filename) as f:
        for line in f:
            attribute_name_and_value_string_list = line.strip().split(':')
            attribute_name = attribute_name_and_value_string_list[0]
            if len(attribute_name_and_value_string_list) < 2:
                attribute_values.append({}) # no values for this attribute
            else:
                value_abbreviation_description_dict = {}
                description_and_abbreviation_string_list = attribute_name_and_value_string_list[1].strip().split(',')
                for description_and_abbreviation_string in description_and_abbreviation_string_list:
                    description_and_abbreviation = description_and_abbreviation_string.strip().split('=')
                    description = description_and_abbreviation[0]
                    if len(description_and_abbreviation) < 2: # assumption: no more than 1 value is missing an abbreviation
                        value_abbreviation_description_dict[None] = description
                    else:
                        abbreviation = description_and_abbreviation[1]
                        value_abbreviation_description_dict[abbreviation] = description
                attribute_values.append(value_abbreviation_description_dict)
    return attribute_values


def load_attribute_names_and_values(filename):
    '''Returns a list of attribute names and values in filename.
    
    This list contains dictionaries wherein the keys are names 
    and the values are value description dictionaries.
    
    Each value description sub-dictionary will use the attribute value abbreviations as its keys 
    and the attribute descriptions as the values.
    
    filename is expected to have one attribute name and set of values per line, with the following format:
        name: value_description=value_abbreviation[,value_description=value_abbreviation]*
    for example
        cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
    The attribute name and values dictionary created from this line would be the following:
        {'name': 'cap-shape', 'values': {'c': 'conical', 'b': 'bell', 'f': 'flat', 'k': 'knobbed', 's': 'sunken', 'x': 'convex'}}'''
    attribute_names_and_values = [] # this will be a list of dicts
    with open(filename) as f:
        for line in f:
            attribute_name_and_value_dict = {}
            attribute_name_and_value_string_list = line.strip().split(':')
            attribute_name = attribute_name_and_value_string_list[0]
            attribute_name_and_value_dict['name'] = attribute_name
            if len(attribute_name_and_value_string_list) < 2:
                attribute_name_and_value_dict['values'] = None # no values for this attribute
            else:
                value_abbreviation_description_dict = {}
                description_and_abbreviation_string_list = attribute_name_and_value_string_list[1].strip().split(',')
                for description_and_abbreviation_string in description_and_abbreviation_string_list:
                    description_and_abbreviation = description_and_abbreviation_string.strip().split('=')
                    description = description_and_abbreviation[0]
                    if len(description_and_abbreviation) < 2: # assumption: no more than 1 value is missing an abbreviation
                        value_abbreviation_description_dict[None] = description
                    else:
                        abbreviation = description_and_abbreviation[1]
                        value_abbreviation_description_dict[abbreviation] = description
                attribute_name_and_value_dict['values'] = value_abbreviation_description_dict
            attribute_names_and_values.append(attribute_name_and_value_dict)
    return attribute_names_and_values
    
    
def attribute_values(instances, attribute_index):
    '''Returns the distinct values of an attribute across a list of instances.
    
    instances is expected to be a list of instances (attribute values).
    attribute_index is expected bo be a the position of attribute in instances.
    
    See http://www.peterbe.com/plog/uniqifiers-benchmark for variants on this algorirthm'''
    return list(set([x[attribute_index] for x in instances]))


def attribute_value(instance, attribute, attribute_names):
    '''Returns the value of an attribute in an instance.
    
    Based on the position of attribute in the list of attribute_names'''
    if attribute not in attribute_names:
        return None
    else:
        i = attribute_names.index(attribute)
        return instance[i] # using the parameter name here
        

def print_attribute_names_and_values(instance, attribute_names):
    '''Prints the attribute names and values for instance'''
    print('Values for the', len(attribute_names), 'attributes:', end='\n\n')
    for i in range(len(attribute_names)):
        print(attribute_names[i], '=', 
        	  attribute_value(instance, attribute_names[i], attribute_names))


def attribute_value_counts(instances, attribute, attribute_names):
    '''Returns a Counter containing the counts of occurrences
     of each value of attribute in the list of instances.
    attribute_names is a list of names of attributes.'''
    i = attribute_names.index(attribute)
    return Counter([instance[i] for instance in instances])


def print_all_attribute_value_counts(instances, attribute_names):
    '''Returns a list of Counters containing the counts of occurrences 
    of each value of each attribute in the list of instances.
    attribute_names is a list of names of attributes.'''
    num_instances = len(instances)
    for attribute in attribute_names:
        value_counts = attribute_value_counts(instances, attribute, attribute_names)
        print('{}:'.format(attribute), end=' ')
        for value, count in sorted(value_counts.items(), key=operator.itemgetter(1), reverse=True):
            print('{} = {} ({:5.3f}),'.format(value, count, count / num_instances), end=' ')
        print()
        
    
def entropy(instances, class_index=0, attribute_name=None, value_name=None):
    '''Calculate the entropy of attribute in position attribute_index for the list of instances.'''
    num_instances = len(instances)
    if num_instances <= 1:
        return 0
    value_counts = defaultdict(int)
    for instance in instances:
        value_counts[instance[class_index]] += 1
    num_values = len(value_counts)
    if num_values <= 1:
        return 0
    attribute_entropy = 0.0
    if attribute_name:
        print('entropy({}{}) = '.format(attribute_name, 
        	'={}'.format(value_name) if value_name else ''))
    for value in value_counts:
        value_probability = value_counts[value] / num_instances
        child_entropy = value_probability * math.log(value_probability, 2)
        attribute_entropy -= child_entropy
        if attribute_name:
            print('  - p({0}) x log(p({0}), {1})  =  - {2:5.3f} x log({2:5.3f})  =  {3:5.3f}'.format(
                value, num_values, value_probability, child_entropy))
    if attribute_name:
        print('  = {:5.3f}'.format(attribute_entropy))
    return attribute_entropy


def information_gain(instances, parent_index, class_index=0, attribute_name=False):
    '''Return the information gain of splitting the instances based on the attribute parent_index'''
    parent_entropy = entropy(instances, class_index, attribute_name)
    child_instances = defaultdict(list)
    for instance in instances:
        child_instances[instance[parent_index]].append(instance)
    children_entropy = 0.0
    num_instances = len(instances)
    for child_value in child_instances:
        child_probability = len(child_instances[child_value]) / num_instances
        children_entropy += child_probability * entropy(
        	child_instances[child_value], class_index, attribute_name, child_value)
    return parent_entropy - children_entropy
    

def majority_value(instances, class_index=0):
    '''Return the most frequent value of class_index in instances'''
    class_counts = Counter([instance[class_index] for instance in instances])
    return class_counts.most_common(1)[0][0]


def choose_best_attribute_index(instances, candidate_attribute_indexes, class_index=0):
    '''Return the index of the attribute that will provide the greatest information gain 
    if instances were partitioned based on that attribute'''
    gains_and_indexes = sorted([(information_gain(instances, i), i) for i in candidate_attribute_indexes], 
                               reverse=True)
    return gains_and_indexes[0][1]


def cmp_partitions(p1, p2):
    if entropy(p1) < entropy(p2):
        return -1
    elif entropy(p1) > entropy(p2):
        return 1
    elif len(p1) < len(p2):
        return -1
    elif len(p1) > len(p2):
        return 1
    return 0


def split_instances(instances, attribute_index):
    '''Returns a list of dictionaries, splitting a list of instances according to their values of a specified attribute''
    
    The key of each dictionary is a distinct value of attribute_index,
    and the value of each dictionary is a list representing the subset of instances that have that value for the attribute'''
    partitions = defaultdict(list)
    for instance in instances:
        partitions[instance[attribute_index]].append(instance)
    return partitions


def partition_instances(instances, num_partitions):
    '''Returns a list of relatively equally sized disjoint sublists (partitions) of the list of instances'''
    return [[instances[j] for j in range(i, len(instances), num_partitions)] for i in xrange(num_partitions)]


def create_decision_tree(instances, candidate_attribute_indexes=None, class_index=0, default_class=None, trace=0):
    '''Returns a new decision tree trained on a list of instances.
    
    The tree is constructed by recursively selecting and splitting instances based on 
    the highest information_gain of the candidate_attribute_indexes.
    
    The class label is found in position class_index.
    
    The default_class is the majority value for the current node's parent in the tree.
    A positive (int) trace value will generate trace information with increasing levels of indentation.
    
    Derived from the simplified ID3 algorithm presented in Building Decision Trees in Python by Christopher Roach,
    http://www.onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html?page=3'''
    
    # if no candidate_attribute_indexes are provided, assume that we will use all but the target_attribute_index
    if candidate_attribute_indexes is None:
        candidate_attribute_indexes = [i for i in range(len(instances[0])) if i != class_index]
        #candidate_attribute_indexes.remove(class_index)
        
    class_labels_and_counts = Counter([instance[class_index] for instance in instances])

    # If the dataset is empty or the candidate attributes list is empty, return the default value
    if not instances or not candidate_attribute_indexes:
        if trace:
            print('{}Using default class {}'.format('< ' * trace, default_class))
        return default_class
    
    # If all the instances have the same class label, return that class label
    elif len(class_labels_and_counts) == 1:
        class_label = class_labels_and_counts.most_common(1)[0][0]
        if trace:
            print('{}All {} instances have label {}'.format('< ' * trace, 
            	len(instances), class_label))
        return class_label
    else:
        default_class = simple_ml.majority_value(instances, class_index)

        # Choose the next best attribute index to best classify the instances
        best_index = simple_ml.choose_best_attribute_index(instances, candidate_attribute_indexes, class_index)        
        if trace:
            print('{}Creating tree node for attribute index {}'.format('> ' * trace, best_index))

        # Create a new decision tree node with the best attribute index and an empty dictionary object (for now)
        tree = {best_index:{}}

        # Create a new decision tree sub-node (branch) for each of the values in the best attribute field
        partitions = simple_ml.split_instances(instances, best_index)

        # Remove that attribute from the set of candidates for further splits
        remaining_candidate_attribute_indexes = [i for i in candidate_attribute_indexes if i != best_index]
        for attribute_value in partitions:
            if trace:
                print('{}Creating subtree for value {} ({}, {}, {}, {})'.format(
                    '> ' * trace,
                    attribute_value, 
                    len(partitions[attribute_value]), 
                    len(remaining_candidate_attribute_indexes), 
                    class_index, 
                    default_class))
                
            # Create a subtree for each value of the the best attribute
            subtree = create_decision_tree(
                partitions[attribute_value],
                remaining_candidate_attribute_indexes,
                class_index,
                default_class,
                trace + 1 if trace else 0)

            # Add the new subtree to the empty dictionary object in the new tree/node we just created
            tree[best_index][attribute_value] = subtree

    return tree


def classify(tree, instance, default_class=None):
    '''Returns a classification label for instance, given a decision tree'''
    if not tree:
        return default_class
    if not isinstance(tree, dict): 
        return tree
    attribute_index = list(tree.keys())[0]  # using list(dict.keys()) for Python 3 compatibiity
    attribute_values = list(tree.values())[0]
    instance_attribute_value = instance[attribute_index]
    if instance_attribute_value not in attribute_values:
        return default_class
    return classify(attribute_values[instance_attribute_value], instance, default_class)


def classification_accuracy(tree, testing_instances, class_index=0):
    '''Returns the accuracy of classifying testing_instances with tree, 
    where the class label is in position class_index'''
    num_correct = 0
    for i in range(len(testing_instances)):
        prediction = classify(tree, testing_instances[i])
        actual_value = testing_instances[i][class_index]
        if prediction == actual_value:
            num_correct += 1
    return num_correct / len(testing_instances)
    

def compute_learning_curve(instances, num_partitions=10):
    '''Returns a list of training sizes and scores for incrementally increasing partitions.
    
    The list contains 2-element tuples, each representing a training size and score.
    The i-th training size is the number of instances in partitions 0 through num_partitions - 2.
    The i-th score is the accuracy of a tree trained with instances 
    from partitions 0 through num_partitions - 2
    and tested on instances from num_partitions - 1 (the last partition).'''
    partitions = partition_instances(instances, num_partitions)
    testing_instances = partitions[-1][:]
    training_instances = partitions[0][:]
    accuracy_list = []
    for i in range(1, num_partitions):
        # for each iteration, the training set is composed of partitions 0 through i - 1
        tree = create_decision_tree(training_instances)
        partition_accuracy = classification_accuracy(tree, testing_instances)
        accuracy_list.append((len(training_instances), partition_accuracy))
        training_instances.extend(partitions[i][:])
    return accuracy_list
