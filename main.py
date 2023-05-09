from parse_data import get_train, get_test
from imputation import mean_imputation
from submit import submit
from libsvm.svmutil import *

train_y, train_x = get_train()

train_x = mean_imputation(train_x)

m = svm_train(train_y, train_x, "-s 0 -c 1 -e 0.00001")

p_labels, p_acc, p_vals = svm_predict(train_y, train_x, m)

test_x = get_test()
test_x = mean_imputation(test_x)

p_labels, p_acc, p_vals = svm_predict([], test_x, m)

submit(p_labels)