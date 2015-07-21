from __future__ import print_function, division

''' A class to implement a simple decision tree (based on ID3)
'''

__author__ = 'Joe McCarthy'
__email__ = 'joe@interrelativity.com'


from collections import Counter
from pprint import pprint
from simple_ml import majority_value, choose_best_attribute_index, split_instances


class SimpleDecisionTree:


    _tree = {}  # this instance variable becomes accessible to class methods via self._tree


    def __init__(self):
        # this is where we would initialize any parameters to the SimpleDecisionTree
        pass
            
    def fit(self, 
            instances, 
            candidate_attribute_indexes=None,
            target_attribute_index=0,
            default_class=None,
            trace=0):
        '''
        Build a decision tree that best fits the data in instances.
        
        The target_attribute_index defaults to 0 (zero).
        The candidate_attribute_indexes defaults to all other index values.

        The tree is constructed by recursively selecting & splitting instances based on 
        the highest information_gain of the candidate_attribute_indexes.
        The class label is found in target_attribute_index.
        The default_class is the majority value for that branch of the tree.
        A positive trace value will print trace information during tree construction.
    
        Derived from the simplified ID3 algorithm presented in 
        Building Decision Trees in Python by Christopher Roach,
        http://www.onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html?page=3
        '''
        if not candidate_attribute_indexes:
            candidate_attribute_indexes = [i 
                                           for i in range(len(instances[0]))
                                           if i != target_attribute_index]
        self._tree = self._create_tree(instances,
                                       candidate_attribute_indexes,
                                       target_attribute_index,
                                       default_class)


    def _create_tree(self,
                     instances,
                     candidate_attribute_indexes,
                     target_attribute_index=0,
                     default_class=None,
                     trace=0):
        class_labels_and_counts = Counter([instance[target_attribute_index] 
                                           for instance in instances])
        # If the dataset is empty or the candidate attributes list is empty, 
        # return the default class label
        if not instances or not candidate_attribute_indexes:
            if trace:
                print('{}Using default class {}'.format('< ' * trace, default_class))
            return default_class
            
        # If all the instances have the same class label, return that class label
        elif len(class_labels_and_counts) == 1:
            class_label = class_labels_and_counts.most_common(1)[0][0]
            if trace:
                print('{}All {} instances have label {}'.format(
                    '< ' * trace, len(instances), class_label))
            return class_label

		# Otherwise, create a new subtree and add it to the tree
        else:
            default_class = majority_value(instances, target_attribute_index)

            # Choose the next best attribute index to best classify the instances
            best_index = choose_best_attribute_index(instances,
                                                     candidate_attribute_indexes, 
                                                     target_attribute_index)
            if trace:
                print('{}Creating tree node for attribute index {}'.format(
                	'> ' * trace, best_index))

            # Create a new decision tree node with the best attribute index 
            # and an empty dictionary object (for now)
            tree = {best_index:{}}

            # Create a new decision tree sub-node (branch) 
            # for each of the values in the best attribute field       
            partitions = split_instances(instances, best_index)
            
            # Remove that attribute from the set of candidates for further splits
            remaining_candidate_attribute_indexes = [i 
                                                     for i in candidate_attribute_indexes 
                                                     if i != best_index]
                                                     
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
                subtree = self._create_tree(
                    partitions[attribute_value],
                    remaining_candidate_attribute_indexes,
                    target_attribute_index,
                    default_class)

                # Add the new subtree to the empty dictionary object 
                # in the new tree/node created above
                tree[best_index][attribute_value] = subtree

            return tree


    def predict(self, instances, default_class=None):
    	'''Return the predicted class label(s) of instance(s)'''
        if not isinstance(instances, list):
            return self._predict(self._tree, instance, default_class)
        else:
            return [self._predict(self._tree, instance, default_class) 
                    for instance in instances]

        
    # a method intended to be "protected" that can implement the recursive algorithm to classify an instance given a tree
    def _predict(self, tree, instance, default_class=None):
        if not tree:
            return default_class
        if not isinstance(tree, dict):
            return tree
        attribute_index = list(tree.keys())[0]  # using list(dict.keys()) for Py3 compatibiity
        attribute_values = list(tree.values())[0]
        instance_attribute_value = instance[attribute_index]
        if instance_attribute_value not in attribute_values:
            return default_class
        return self._predict(attribute_values[instance_attribute_value],
                             instance,
                             default_class)


    def classification_accuracy(self, instances, default_class=None):
    	'''Return a tuple with 
    	the number of correctly classified instances,
    	the number of incorrectly classified instances,
    	the proportion of instances that were correctly classified
    	'''
        predicted_labels = self.predict(instances, default_class)
        actual_labels = [x[0] for x in instances]
        counts = Counter([x == y for x, y in zip(predicted_labels, actual_labels)])
        return counts[True] / len(instances), counts[True], counts[False]
        
        
    def pprint(self):
        pprint(self._tree)