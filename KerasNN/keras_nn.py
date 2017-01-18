"""
    Based off example here:

    http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


def generate_data():
    """
        Generates a normal distribution outlined here:

        https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivariate_normal.html
    :return: 2 lists of points with their classifications
    """
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    red_x, red_y = np.random.multivariate_normal(mean, cov, 300).T
    red_points = np.column_stack((red_x, red_y))
    red_class = np.ones((len(red_points), 1))
    red_points_class = np.append(red_points, red_class, axis=1)

    mean = [-4, 0]
    blue_left_x, blue_left_y = np.random.multivariate_normal(mean, cov, 300).T
    blue_x = blue_left_x
    blue_y = blue_left_y

    mean = [4, 0]
    blue_right_x, blue_right_y = np.random.multivariate_normal(mean, cov, 300).T
    blue_x = np.append(blue_x, blue_right_x, axis=0)
    blue_y = np.append(blue_y, blue_right_y, axis=0)

    mean = [0, -4]
    blue_bottom_x, blue_bottom_y = np.random.multivariate_normal(mean, cov, 300).T
    blue_x = np.append(blue_x, blue_bottom_x, axis=0)
    blue_y = np.append(blue_y, blue_bottom_y, axis=0)

    blue_points = np.column_stack((blue_x, blue_y))
    blue_class = np.zeros((len(blue_points), 1))
    blue_points_class = np.append(blue_points, blue_class, axis=1)

    return {
        "red_points_class": red_points_class,
        "blue_points_class": blue_points_class
    }


def gen_train_and_test(list1, list2, ratio):
    """

    :param list1:
    :param list2:
    :param ratio: train:test
    :return: a training and test set created from list1 and list2 based on ratio provided
    """
    np.random.shuffle(list1)
    np.random.shuffle(list2)
    list1_index = int(ratio * len(list1))
    list2_index = int(ratio * len(list2))
    train = np.append(list1[0:list1_index], list2[0:list2_index], axis=0)
    test = np.append(list1[list1_index:], list2[list2_index:], axis=0)

    return {
        "train": train,
        "test": test
    }


def create_model():
    model = Sequential()

    # First hidden layer has 3 neurons and 2 input variables
    model.add(Dense(3, input_dim=2, activation='relu'))

    # Output layer has 1 neuron to classify
    model.add(Dense(1, activation='sigmoid'))

    return model

if __name__ == "__main__":
    data = generate_data()
    dataset = gen_train_and_test(data["red_points_class"], data["blue_points_class"], 0.7)
    train = dataset["train"]
    test = dataset["test"]

    # Visualize data points
    # plt.plot(data["red_points_class"][:, 0], data["red_points_class"][:, 1], 'x', c='r')
    # plt.plot(data["blue_points_class"][:, 0], data["blue_points_class"][:, 1], 'x', c='b')
    # plt.axis('equal')
    # plt.show()

    # Create keras model
    model = create_model()

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(train[:, 0:2], train[:, 2], nb_epoch=150, batch_size=10)

    # Evaluate the model
    scores = model.evaluate(test[:, 0:2], test[:, 2])
    """
        The accuracy fluctuates, but usually gets over 95%
        acc: 97.50%
    """
    print("\n\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
