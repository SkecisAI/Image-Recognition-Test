import os
import joblib
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier


s = []
digit_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alpha_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# for file in files:
#     if not os.path.isdir(file):
#         print(type(file))


def digit_train(filename):
    my_mlp = MLPClassifier()
    lab = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_set, y_label = load_dataset(digit_labels, lab)
    model = my_mlp.fit(x_set, y_label)
    save_model(model, 'digit')
    test_sample = Image.open(filename).getdata()
    test_sample = np.array(test_sample, dtype='int').reshape(1, -1)
    ans_prob = model.predict_proba(test_sample).tolist()[0]
    return digit_labels[ans_prob.index(max(ans_prob))]


def alpha_train(filename):
    my_mlp = MLPClassifier()
    lab = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_set, y_label = load_dataset(alpha_labels, lab)
    model = my_mlp.fit(x_set, y_label)
    save_model(model, 'alpha')
    test_sample = Image.open(filename).getdata()
    test_sample = np.array(test_sample, dtype='int').reshape(1, -1)
    ans_prob = model.predict_proba(test_sample).tolist()[0]
    return alpha_labels[ans_prob.index(max(ans_prob))]


def load_dataset(labels_list, lab_v):
    train_x = []
    train_y = []
    i = 0
    for label in labels_list:
        i += 1
        lab_vec = lab_v.copy()
        path = "train_set/"+label
        img_files = os.listdir(path)
        for file in img_files:
            img = Image.open(path+'/'+file)
            img_array = img.getdata()
            img_array = np.array(img_array, dtype='int')
            lab_vec[i-1] = 1
            train_x.append(img_array)
            train_y.append(lab_vec)
    return train_x, train_y


def save_model(model, name):
    joblib.dump(model, name + '.pkl')


pre = digit_train("img_crop_set/459_3.png")
print(pre)
pre = alpha_train("img_crop_set/381_4.png")
print(pre)