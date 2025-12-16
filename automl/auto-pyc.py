import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import os
import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

#train데이터의 id, fateher, mother, gender은 필요없는 변수이므로 제거 test도 마찬가지
train=train.iloc[:,4:]
test=test.iloc[:,4:]

train.head()

#자동으로 label encoding 진행
from pycaret.classification import *

# 1. 데이터 준비
setup_clf = setup(data=train, target='class', train_size=0.8,
                  session_id=777)
# 2. 모델 비교
models()
best_model = compare_models()
# 2. 모델 비교
model = compare_models(sort='F1',fold=3,n_select=5)

# 3. 모델 튜닝 및 앙상블
# 모델 튜닝
tuned_model = [tune_model(i) for i in model]

# 모델 앙상블
#blended_model = blend_models(estimator_list=tuned_model)

tuned_model

blended = blend_models(estimator_list = tuned_model,
                       fold = 10,
                       method = 'soft',
                       optimize='F1',
                       )

# 모델 성능평가
final_model = finalize_model(blended)
evaluate_model(final_model)

from pycaret.utils import check_metric

prediction = predict_model(final_model, data=test_x)

prediction 

submit = pd.read_csv('./data/sample_submission.csv')
submit['class'] = prediction.Label
submit.to_csv('submit_pycaret.csv', index=False)