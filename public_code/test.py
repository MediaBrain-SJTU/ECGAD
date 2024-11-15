
from model_template import InferenceSdk
import pandas as pd
import os


# Single file test
model = InferenceSdk.RatiocinationSdk(gpu_id=-1, weight_dir='./model_template/weights')
results = model.classify(input_json_path='./example/example_1.json')
print(results)

source_folder = 'example/'
# Multiple file test
for file_name in os.listdir(source_folder):
    file_path = os.path.join(source_folder, file_name)
    prediction = model.classify(input_json_path=file_path)
    print("Prediction: ", prediction)