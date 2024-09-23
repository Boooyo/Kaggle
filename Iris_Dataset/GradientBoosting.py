# 필요한 라이브러리 임포트
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터를 학습 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 그래디언트 부스팅 분류기
from sklearn.ensemble import GradientBoostingClassifier

GB_model = GradientBoostingClassifier()
GB_model.fit(X_train, y_train)

y_predict_gb = GB_model.predict(X_test)
print("그래디언트 부스팅 정확도: ", accuracy_score(y_test, y_predict_gb))