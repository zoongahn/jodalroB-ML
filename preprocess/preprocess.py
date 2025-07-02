from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from utils.classify_columns import classify_columns

# 1) 컬럼 분류
meta_df = pd.read_csv('../meta/notice.csv', dtype=str)

column_groups = classify_columns(meta_df)

numeric_cols = column_groups['numeric']
categorical_cols = column_groups['categorical']
text_cols = column_groups['text']

# 2) 각 타입별 서브파이프라인
numeric_pipeline = Pipeline([
	('imputer', SimpleImputer(strategy='median')),  # 중앙값으로 결측치 채움
	('scaler', StandardScaler()),  # 표준화
])

categorical_pipeline = Pipeline([
	('imputer', SimpleImputer(strategy='most_frequent')),  # 최빈값으로 결측치 채움
	('onehot', OneHotEncoder(handle_unknown='ignore')),  # 원-핫 인코딩
])

text_pipeline = Pipeline([
	('tfidf', TfidfVectorizer(
		max_features=2000,  # 최대 2000개 단어만 사용
		ngram_range=(1, 2),  # 1-gram + 2-gram
		stop_words='english'  # 필요에 따라 한국어 불용어 리스트 사용
	))
])

# 3) ColumnTransformer 결합
preprocessor = ColumnTransformer([
	('num', numeric_pipeline, numeric_cols),
	('cat', categorical_pipeline, categorical_cols),
	('txt', text_pipeline, text_cols),
], remainder='drop')  # 나머지 컬럼은 제거

# 4) 전체 파이프라인
pipeline = Pipeline([
	('prep', preprocessor),
	# ('clf', YourModel())    # 예: 분류/회귀 모델 연결
])

# 5) 사용 예시
# X_raw: DataFrame 형태의 원본 데이터
# y: 목표 변수(label)
X_transformed = pipeline.fit_transform(X_raw, y)
print("Shape:", X_transformed.shape)
