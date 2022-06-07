####################
# read csv document
####################

# this program is for label0_data0.csv
# label1_data0.csv need to change document name
f = open('label0_data0.csv', 'r')
lines = f.readlines()
f.close()

data = []
label = []

for line in lines[3:]:
    line = line.replace('\n', '').split(',')
    data.append( list(map(float, line[3:])) ) # change string to float
    label.append( int(line[0]) ) # change string to int

import numpy as np
data = np.array(data)
label = np.array(label)

print('data.shape =', data.shape)
print('label.shape =', label.shape)
print()

####################
# read groups
####################

f = open('groups.csv', 'r')
lines = f.readlines()
f.close()

groups = []

for line in lines[3:]:
    line = line.replace('\n', '').split(',')
    groups.append( int(line[1]) )

groups = np.array(groups)

print('groups.shape =', groups.shape)
print('np.unique(groups) =', np.unique(groups))
print()





####################
# split dataset
# not stable
# should change to k-fold
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# k = 5 or 10
####################

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    data, label,
    test_size = 0.3, random_state = 42,
    stratify = label ) # Asymmetric dataset, split proportionally when splitting

####################
# Group Lasso
####################

from group_lasso import GroupLasso

gl = GroupLasso(
    groups = groups,
    group_reg = 0.001, # 0-5 smaller is better
    l1_reg = 0.001, # 0-5 smaller is better
    # frobenius_lipschitz = True,
    scale_reg = 'inverse_group_size',
    # subsampling_scheme = 1,
    supress_warning = True,
    n_iter = 100000,
    # tol = 1e-5,
    # warm_start = True, # not much different
    )

gl.fit(x_train, y_train)

####################
# predict
####################

y_prob = gl.predict(x_test)

print('y_prob.shape =', y_prob.shape)
print()

####################
# confusion matrix
####################

# if there is error, version problem
# conda install scikit-learn=0.22
from sklearn.metrics._plot.confusion_matrix import confusion_matrix

tn, fp, fn, tp = confusion_matrix( y_test,
                                   y_prob > 0.5,
                                   ).ravel()

print()
print()
print()
print( 'acc =', (tn+tp) / (tn+fp+fn+tp) )
print( 'f1 =', 2*tp / (2*tp+fp+fn) )
print()
        
print('Confusion Matrix')
print('=====================')
print('TN = ' + str(tn), end=' | ')
print('FP = ' + str(fp))
print('---------------------')
print('FN = ' + str(fn), end=' | ')
print('TP = ' + str(tp))
print()

####################
# split to category
####################

y_test_2 = []
for i in y_test:
    y_test_2.append([i, 1-i])
y_test_2 = np.array(y_test_2)

y_prob_2 = []
for i in y_prob:
    y_prob_2.append([i, 1-i])
y_prob_2 = np.array(y_prob_2)

print('y_test_2.shape =', y_test_2.shape)
print('y_prob_2.shape =', y_prob_2.shape)
print()

####################
# auc-roc
####################

fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = y_prob_2.shape[1]
                
from sklearn.metrics import roc_curve, auc

# Calculate each point of the ROC curve using the actual class
# and predicted probability
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve( y_test_2[:, i], y_prob_2[:, i] )
    roc_auc[i] = auc(fpr[i], tpr[i])

print('roc_auc =', roc_auc)
print()

####################
# draw ROC AUC with plt
####################

import matplotlib.pyplot as plt
                
plt.plot( fpr[0], tpr[0], color = 'darkorange',
          lw = 2, label = 'ROC curve (area = %0.6f)' % roc_auc[1])
                
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
                
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc = 'lower right')

plt.show()



####################
# output weight
####################

fw = open('gl_output_weight.csv', 'w')

for i in gl.coef_:

    print(abs(i[0]), file=fw)

fw.close()
