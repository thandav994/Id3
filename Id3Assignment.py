import pandas as pd
import math
from logging import root

baseNode = None

class node:
    def __init__(self):
        self.left = None
        self.right = None
        self.entropy = 0.0
        self.parent = None
        self.label = None

def calculateInformationGain(feature, dataSet, parent):
    if parent is None:
        parentEntropy = calculateEntropy(dataSet, dataSet.get('Class'))
    if parent is not None:
        parentEntropy = calculateEntropy(dataSet, dataSet.get(parent.label))
        parent.entropy = parentEntropy
        
    filteredSet0 = createFilteredSet(dataSet, feature, 0)
    filteredSet1 = createFilteredSet(dataSet, feature, 1)
    entropy0 = calculateEntropy(filteredSet0, filteredSet0.get(feature))
    entropy1 = calculateEntropy(filteredSet1, filteredSet1.get(feature))
    n0 = len(filteredSet0.get(feature))
    n1 = len(filteredSet1.get(feature))
    m = n0 + n1
#     print "n0,n1",n0,n1
#     if parent is not None:
#         print "feature:",feature,"parent:",parent.label,"parent entropy:",parentEntropy,",entropy0:",entropy0,",entropy1:",entropy1
#     print "n0:",n0,",n1",n1
    return parentEntropy - (float(n0)/m)*entropy0 - (float(n1)/m)*entropy1
    
def findBestFeature(filteredSet, parent, features):
    highest_information_gain = -1
    best_feature = None
#     print features
    for feature in features:
        information_gain = calculateInformationGain(feature, filteredSet, parent)
#         print "Feature:",feature," Information gain:",information_gain
        if information_gain > highest_information_gain:
            highest_information_gain = information_gain
            best_feature = feature
    bestNode = node()
    bestNode.parent = parent
    bestNode.label = best_feature 
#     print "Best Node:",bestNode.label
    if highest_information_gain in [0,0.0] :
        return None
    return bestNode
    
def isPureData(dataSet):
    isSame = True
    first = dataSet.get('Class')[0]
    for each in dataSet.get('Class'):
        if each != first:
            isSame = False
            break
    return isSame
    
def createFilteredSet(dataSet, feature, value):
    indices = [i for i, x in enumerate(dataSet.get(feature)) if x == value]
#     print "dataset:",dataSet
#     print "feature:",feature
#     print "value:", value
    filteredSet = {}
    for each in dataSet.keys():
        values = dataSet.get(each)
        filtered_values = []
        for index in indices:
            filtered_values.append(values[index])
        filteredSet.update({each: filtered_values})
#     print "filtered set:",filteredSet.get(feature)
    return filteredSet
        
def createDecisionTree(dataSet,rootNode, features):
    # print baseEntropy
    # End of recursion
    if isPureData(dataSet) or not features:
        leaf_node = node()
        leaf_node.label= dataSet.get('Class')[0] 
        if leaf_node.label == 0 :
            rootNode.right = leaf_node
            leaf_node.parent = rootNode
            print "LEAF NODE on RIGHT!!! : ",rootNode.right.label
        if leaf_node.label == 1 :
            rootNode.left = leaf_node
            leaf_node.parent = rootNode
            print "LEAF NODE on LEFT!!! : ",rootNode.left.label
        return
     
    if rootNode is None:
        best_feature_Node = findBestFeature(dataSet,rootNode, features)
        if best_feature_Node is None:
            return
        rootNode = best_feature_Node
        global baseNode
        baseNode = rootNode
        
    possibleValues = [0,1]
    # Setting left and right trees
    for each in possibleValues:
        remaining_features = features[:]
        remaining_features.remove(rootNode.label)
#         print "dataSet",dataSet
        filteredSet = createFilteredSet(dataSet, rootNode.label, each)
        print "rootNode:",rootNode.label
#         print "filtered Set",filteredSet
        if len(filteredSet.get('Class')) is 0:
            createDecisionTree(dataSet, rootNode, remaining_features)
        else:
            best_feature_Node = findBestFeature(filteredSet, rootNode, remaining_features)
            if best_feature_Node is None:
                print "Information gain is zero"
                createDecisionTree(filteredSet, rootNode, remaining_features)
            else:
                if each is 0:
                    rootNode.right = best_feature_Node
                if each is 1:
                    rootNode.left = best_feature_Node
                best_feature_Node.parent = rootNode
                createDecisionTree(filteredSet, best_feature_Node, remaining_features)
        
            
def calculateEntropy(dataSet, values):
    class_values = dataSet.get('Class')
    y = 0.0 
    n = 0.0
    for index in range(0,len(values)):
        if class_values[index] == 0:
            n = n+1
        else:
            y = y+1
    m = y+n
    if m is 0.0 or n is 0.0 or y is 0.0:
        return 0
    return -((n/m)*math.log(n/m, 2) + (y/m)*math.log(y/m, 2))


def printTree(rootNode,depth):
    if rootNode is not None:
        if rootNode.left is None and rootNode.right is None:
            print rootNode.label
        else:
            print rootNode.label,'= 1:'
            for x in range(0,depth):
                print('|'),
            printTree(rootNode.left,depth+1)
            print ""
            print rootNode.label,'= 0:'
            for x in range(0,depth):
                print('|'),
            if rootNode.left is None and rootNode.right is None:
                depth = 0
                return
            printTree(rootNode.right,depth+1)

dataSet = {}
headers = []
df = pd.read_csv('training_set1.csv')
for each in df.columns:
    headers.append(each)
for each in headers:
    dataSet.update({each : df.get(each)})
features = dataSet.keys()
features.remove('Class')
createDecisionTree(dataSet,None, features)
printTree(baseNode,1)
print baseNode.left.left.left.left.left.right.label
print "Base Node:",baseNode.label
print "right:",baseNode.right.label
print "left:",baseNode.left.label
# print "right.right:",baseNode.right.right.label
# print "right.left:",baseNode.right.left.label
# print "left.right:",baseNode.right.right.left.right.label
# print "left.right:",baseNode.right.right.left.left.label