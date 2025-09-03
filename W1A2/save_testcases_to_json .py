import numpy as np
import json
from testCases import compute_cost_with_regularization_test_case

# 1. 테스트 케이스 가져오기
a3, Y, parameters = compute_cost_with_regularization_test_case()

# 2. NumPy 배열을 리스트로 변환
json_data = {
    "a3": a3.tolist(),
    "Y": Y.tolist(),
    "parameters": {
        "W1": parameters["W1"].tolist(),
        "b1": parameters["b1"].tolist(),
        "W2": parameters["W2"].tolist(),
        "b2": parameters["b2"].tolist(),
        "W3": parameters["W3"].tolist(),
        "b3": parameters["b3"].tolist()
    }
}

# 3. JSON 파일로 저장
with open("compute_cost_with_regularization_test_case.json", "w") as f:
    json.dump(json_data, f, indent=4)
