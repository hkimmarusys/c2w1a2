import json

with open("2D_dataset.json", "r") as f:
    data = json.load(f)

print("keys:", data.keys())
print("len(train_X):", len(data["train_X"]))
print("len(train_Y):", len(data["train_Y"]))
