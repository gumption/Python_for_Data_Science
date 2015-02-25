from __future__ import print_function, division

''' A class to implement a simple decision tree (based on ID3)
'''

__author__ = 'Joe McCarthy'
__email__ = 'joe@interrelativity.com'


from collections import Counter
from pprint import pprint
import simple_ml


class SimpleDecisionTree:


    _tree = {} # this instance variable becomes accessible to class methods via self._tree


    def __init__(self, instances=None, target_attribute_index=0, trace=0): # note the use of self as the first parameter
        if instances:
            self._tree = self._create(instances, range(1, len(instances[0])), target_attribute_index, trace=trace)


    def _create(self, instances, candidate_attribute_indexes, target_attribute_index=0, default_class=None, trace=0):
        '''
        Returns a new decision tree by recursively selecting and splitting instances based on 
        the highest information_gain of the candidate_attribute_indexes.
        The class label is found in target_attribute_index.
        The default class is the majority value for that branch of the tree.
        A positive trace value will generate trace information with increasing levels of indentation.
    
        Derived from the simplified ID3 algorithm presented in Building Decision Trees in Python by Christopher Roach,
        http://www.onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html?page=3
        '''
        instances = instances[:]
        class_labels_and_counts = Counter([instance[target_attribute_index] for instance in instances])

        # If the dataset is empty or the candidate attributes list is empty, return the default value. 
        if not instances or not candidate_attribute_indexes:
            if trace:
                print('{}Using default class {}'.format('< ' * trace, default_class))
            return default_class

        # If all the instances have the same class label, return that class label
        elif len(class_labels_and_counts) == 1:
            class_label = class_labels_and_counts.most_common(1)[0][0]
            if trace:
                print('{}All {} instances have label {}'.format('< ' * trace, len(instances), class_label))
            return class_label
        else:
            default_class = simple_ml.majority_value(instances, target_attribute_index)

            # Choose the next best attribute index to best classify the instances
            best_index = simple_ml.choose_best_attribute_index(instances, candidate_attribute_indexes, target_attribute_index)
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
                        target_attribute_index, 
                        default_class))

                # Create a subtree for each value of the the best attribute
                subtree = self._create(
                    partitions[attribute_value],
                    remaining_candidate_attribute_indexes,
                    target_attribute_index,
                    default_class,
                    trace + 1 if trace else 0)

                # Add the new subtree to the empty dictionary object in the new tree/node we just created
                tree[best_index][attribute_value] = subtree

        return tree


    # call the internal 'protected' method to classify the instance given the _tree
    def classify(self, instance, default_class=None):   
        return self._classify(self._tree, instance, default_class)

        
    # a method intended to be "protected" that can implement the recursive algorithm to classify an instance given a tree
    def _classify(self, tree, instance, default_class=None):
        if not tree:
            return default_class
        if not isinstance(tree, dict):
            return tree
        attribute_index = list(tree.keys())[0]
        attribute_values = list(tree.values())[0]
        instance_attribute_value = instance[attribute_index]
        if instance_attribute_value not in attribute_values:
            return default_class
        return self._classify(attribute_values[instance_attribute_value], instance, default_class)


    def classify_list(self, instances, default_class=None):
        return [self._classify(self._tree, instance, default_class) for instance in instances]


    def evaluate_accuracy(self, instances, default_class=None):
        predicted_labels = self.classify_list(instances, default_class)
        actual_labels = [x[0] for x in instances]
        counts = Counter([x == y for x, y in zip(predicted_labels, actual_labels)])
        return counts[True], counts[False], counts[True] / len(instances)
        
        
    def pprint(self):
        pprint(self._tree)