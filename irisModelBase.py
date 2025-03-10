# 2025.3.10.
# 프로젝트2 붓꽃분류기 만들기
# 용희가 만
from operator import irshift

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

iris_df = pd.read_csv('iris.csv')

y=iris_df['species']
x= iris_df.drop('species', axis=1)

kn = KNeighborsClassifier()
model_kn = kn.fit(x,y)

#x_new = np.array([[3,3,3,3]])
x_new = np.array([[5.0,3.4,1.4,0.2]])
# ['setosa'] 3 ([[1. 0. 0,]])
# kn ['versicolor'][[[0.0.8 0.2]]
# kn['versicolor']([[1,4.2,1.4,7]])
prediction = model_kn.predict(x_new)
print(prediction)
probability = model_kn.predict_proba(x_new)
print(probability)

