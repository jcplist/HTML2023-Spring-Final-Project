from parse_data import get_train, get_test
from imputation import mean_imputation
from submit import submit
from libsvm.svmutil import *

#used_entry = ["Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms", "Views", "Likes", "Stream", "Comments"]

used_entry = ["Key", "Energy", "Tempo", "Valence", "Speechiness", "Acousticness"]

train_y, train_x = get_train(used_entry)

train_x = mean_imputation(train_x)

m = svm_train(train_y, train_x, "-s 3 -c 10 -e 0.00001")

p_labels, p_acc, p_vals = svm_predict(train_y, train_x, m)

test_x = get_test(used_entry)
test_x = mean_imputation(test_x)

p_labels, p_acc, p_vals = svm_predict([], test_x, m)

submit(p_labels)