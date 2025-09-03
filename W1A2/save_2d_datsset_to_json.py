import json
import scipy.io

data = scipy.io.loadmat('datasets/data.mat')
train_X = data['X'].T
train_Y = data['y'].T
test_X = data['Xval'].T
test_Y = data['yval'].T

json_data = {
    "train_X": train_X.tolist(),
    "train_Y": train_Y.tolist(),
    "test_X": test_X.tolist(),
    "test_Y": test_Y.tolist()
}


with open("2D_dataset.json", "w") as f:
    json.dump(json_data, f, indent=4)

