import pandas as pd
import os
from mimic3benchmarks.mimic3models.preprocessing import Discretizer
import numpy as np
from tqdm import tqdm

dir_path  = '/data1/r10user2/EHR_dataset/mimiciv_benchmark/survival_prediction'

train_path = os.path.join(dir_path , 'train_listfile.csv')
test_path = os.path.join(dir_path , 'test_listfile.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


for patient_path in tqdm(train_df['stay']):
    x_train = []
    patient_path = os.path.join(dir_path, 'train', patient_path)
    patient = pd.read_csv(patient_path)
    patient = patient.fillna('')
    data_processor = Discretizer(impute_strategy='normal_value')
    data = data_processor.transform(np.array(patient))[0]
    #print(data)
    
    print(i)
    i += 1
    x_train.append(data)
    # x_train = np.array(x_train)

print(x_train[0])
