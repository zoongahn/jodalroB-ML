import re
import pandas as pd
from konlpy.tag import Okt, Mecab, Komoran
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class KoreanTextPreprocessor(BaseEstimator, TransformerMixin):
	def __init__(self, tokenizer='okt', remove_stopwords=True, min_length=2):
		self.tokenizer_name = tokenizer
		self.remove_stopwords = remove_stopwords
		self.min_length = min_length

		# 토크나이저 초기화
		if tokenizer == 'okt':
			self.tokenizer = Okt()
		elif tokenizer == 'mecab':
			self.tokenizer = Mecab()
		elif tokenizer == 'komoran':
			self.tokenizer = Komoran()

		# 한국어 불용어 리스트
		self.stopwords = {
			'은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '로', '으로',
			'에서', '까지', '부터', '한테', '에게', '께', '한', '두', '세', '네', '다섯',
			'여섯', '일곱', '여덟', '아홉', '열', '수', '있', '없', '되', '될', '되어',
			'하', '해', '했', '할', '함', '합', '것', '거', '게', '지', '죠', '요', '야',
			'이다', '아니다', '그리고', '그런데', '하지만', '그러나', '또한', '또는'
		}

	def clean_text(self, text):
		"""기본 텍스트 정제"""
		if pd.isna(text):
			return ""

		text = str(text)

		# HTML 태그 제거
		text = re.sub(r'<[^>]+>', '', text)

		# URL 제거
		text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

		# 이메일 제거
		text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)

		# 특수문자 제거 (한글, 영문, 숫자, 공백만 남김)
		text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)

		# 연속된 공백을 하나로
		text = re.sub(r'\s+', ' ', text)

		return text.strip()

	def tokenize_and_pos(self, text):
		"""형태소 분석 및 토큰화"""
		if not text:
			return []

		# 형태소 분석
		if self.tokenizer_name == 'okt':
			tokens = self.tokenizer.pos(text, stem=True)  # 어간 추출
		else:
			tokens = self.tokenizer.pos(text)

		# 명사, 동사, 형용사만 추출
		useful_pos = ['Noun', 'Verb', 'Adjective', 'VV', 'VA', 'NNG', 'NNP']
		filtered_tokens = []

		for word, pos in tokens:
			if any(p in pos for p in useful_pos):
				if len(word) >= self.min_length:
					if not self.remove_stopwords or word not in self.stopwords:
						filtered_tokens.append(word)

		return filtered_tokens

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		"""텍스트 전처리 수행"""
		if isinstance(X, pd.Series):
			X = X.values

		processed_texts = []
		for text in X:
			# 텍스트 정제
			clean = self.clean_text(text)
			# 토큰화
			tokens = self.tokenize_and_pos(clean)
			# 공백으로 연결
			processed_texts.append(' '.join(tokens))

		return processed_texts


# 사용 예시
korean_text_pipeline = Pipeline([
	('preprocess', KoreanTextPreprocessor(tokenizer='okt', remove_stopwords=True)),
	('tfidf', TfidfVectorizer(
		max_features=5000,
		ngram_range=(1, 2),
		min_df=2,
		max_df=0.95,
		sublinear_tf=True
	))
])
