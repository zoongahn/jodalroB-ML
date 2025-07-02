import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import re
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
	"""분석 설정을 위한 데이터클래스"""
	max_features: int = 5000
	ngram_range: Tuple[int, int] = (1, 2)
	min_df: int = 2
	max_df: float = 0.95
	min_word_length: int = 2
	max_word_length: int = 50
	high_tfidf_threshold: float = 0.1
	stopword_doc_freq_threshold: float = 0.6
	stopword_tfidf_threshold: float = 0.02


class TextPreprocessor:
	"""텍스트 전처리 클래스"""

	def __init__(self, remove_numbers: bool = False, remove_special: bool = True):
		self.remove_numbers = remove_numbers
		self.remove_special = remove_special

	def clean_text(self, text: str) -> str:
		"""텍스트 정리"""
		if pd.isna(text) or text is None:
			return ""

		text = str(text).strip()

		if self.remove_special:
			# 특수문자 제거 (단, 한글, 영문, 숫자, 공백은 유지)
			text = re.sub(r'[^\w\s가-힣]', ' ', text)

		if self.remove_numbers:
			text = re.sub(r'\d+', '', text)

		# 연속된 공백 제거
		text = re.sub(r'\s+', ' ', text)

		return text.strip()

	def tokenize(self, text: str) -> List[str]:
		"""단어 토큰화"""
		cleaned = self.clean_text(text)
		tokens = cleaned.split()

		# 길이 필터링
		tokens = [token for token in tokens if 1 <= len(token) <= 50]

		return tokens


class DatabaseTextAnalyzer:
	def __init__(self, db_connection, config: Optional[AnalysisConfig] = None):
		self.db_connection = db_connection
		self.config = config or AnalysisConfig()
		self.preprocessor = TextPreprocessor()
		self.analyzers = {}  # 컬럼별 분석기 저장
		self.results = {}  # 컬럼별 분석 결과 저장
		self.metadata = {}  # 분석 메타데이터

	def get_table_info(self, table_name: str) -> Dict:
		"""테이블 정보 조회"""
		try:
			# 테이블 컬럼 정보
			cursor = self.db_connection.cursor()
			cursor.execute(f"PRAGMA table_info({table_name})")
			columns_info = cursor.fetchall()

			# 테이블 행 수
			cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
			row_count = cursor.fetchone()[0]

			return {
				'columns': [col[1] for col in columns_info],  # 컬럼명만 추출
				'row_count': row_count,
				'column_details': columns_info
			}
		except Exception as e:
			logger.error(f"테이블 정보 조회 실패: {e}")
			return {}

	def load_text_columns(self, table_name: str, text_columns: List[str],
	                      target_column: str = None, limit: int = None,
	                      sample_random: bool = False) -> Dict[str, pd.DataFrame]:
		"""DB에서 텍스트 컬럼들을 로드 (개선된 버전)"""

		# 테이블 정보 확인
		table_info = self.get_table_info(table_name)
		if not table_info:
			logger.error(f"테이블 {table_name} 정보를 가져올 수 없습니다.")
			return {}

		# 컬럼 존재 확인
		available_columns = table_info['columns']
		missing_columns = [col for col in text_columns if col not in available_columns]
		if missing_columns:
			logger.warning(f"존재하지 않는 컬럼: {missing_columns}")
			text_columns = [col for col in text_columns if col in available_columns]

		if target_column and target_column not in available_columns:
			logger.warning(f"타겟 컬럼 {target_column}이 존재하지 않습니다.")
			target_column = None

		# SQL 쿼리 구성
		columns_str = ", ".join([f'"{col}"' for col in text_columns])
		if target_column:
			columns_str += f', "{target_column}"'

		query = f'SELECT {columns_str} FROM "{table_name}"'

		# 샘플링 옵션
		if sample_random and limit:
			query += f' ORDER BY RANDOM() LIMIT {limit}'
		elif limit:
			query += f' LIMIT {limit}'

		logger.info(f"실행 쿼리: {query}")

		try:
			# 데이터 로드
			df = pd.read_sql_query(query, self.db_connection)
			logger.info(f"로드된 데이터: {len(df)}행")

			# 텍스트 컬럼별로 데이터 분리 및 전처리
			column_data = {}
			for col in text_columns:
				# NULL 값 제거 및 문자열 변환
				clean_data = df[col].dropna().astype(str)

				# 빈 문자열 및 공백만 있는 문자열 제거
				clean_data = clean_data[clean_data.str.strip() != '']

				# 텍스트 전처리
				processed_texts = [self.preprocessor.clean_text(text) for text in clean_data]
				processed_texts = [text for text in processed_texts if text]  # 빈 텍스트 제거

				if not processed_texts:
					logger.warning(f"컬럼 {col}: 전처리 후 텍스트가 없습니다.")
					continue

				# 타겟 컬럼이 있으면 함께 저장
				if target_column and target_column in df.columns:
					target_values = df.loc[clean_data.index, target_column]
					# 전처리로 인해 제거된 행들을 고려하여 타겟값도 조정
					valid_indices = [i for i, text in enumerate(processed_texts) if text]
					target_values = [target_values.iloc[i] for i in valid_indices]

					column_data[col] = {
						'texts': processed_texts,
						'targets': target_values
					}
				else:
					column_data[col] = {
						'texts': processed_texts,
						'targets': None
					}

			return column_data

		except Exception as e:
			logger.error(f"데이터 로드 실패: {e}")
			return {}

	def analyze_column(self, column_name: str, texts: List[str], targets: List = None) -> Optional[pd.DataFrame]:
		"""특정 컬럼에 대한 단어 중요도 분석 (개선된 버전)"""

		logger.info(f"\n=== {column_name} 컬럼 분석 시작 ===")
		logger.info(f"총 {len(texts)}개 문서 분석")

		if len(texts) == 0:
			logger.warning(f"컬럼 {column_name}: 분석할 텍스트가 없습니다.")
			return None

		# 1) 전체 단어 빈도 계산
		all_words = []
		doc_word_counts = []

		for text in texts:
			words = self.preprocessor.tokenize(text)
			all_words.extend(words)
			doc_word_counts.append(len(words))

		if not all_words:
			logger.warning(f"컬럼 {column_name}: 토큰화된 단어가 없습니다.")
			return None

		word_freq = Counter(all_words)
		logger.info(f"고유 단어 수: {len(word_freq)}")
		logger.info(f"총 단어 수: {len(all_words)}")
		logger.info(f"평균 문서 단어 수: {np.mean(doc_word_counts):.2f}")

		# 2) TF-IDF 분석
		try:
			# TF-IDF 파라미터 동적 조정
			max_features = min(self.config.max_features, len(word_freq))
			min_df = max(2, min(self.config.min_df, len(texts) // 100))

			vectorizer = TfidfVectorizer(
				max_features=max_features,
				ngram_range=self.config.ngram_range,
				min_df=min_df,
				max_df=self.config.max_df,
				token_pattern=r'\b\w+\b'  # 단어 경계 명확히
			)

			tfidf_matrix = vectorizer.fit_transform(texts)
			feature_names = vectorizer.get_feature_names_out()

			# 평균 TF-IDF 점수
			mean_tfidf = tfidf_matrix.mean(axis=0).A1
			max_tfidf = tfidf_matrix.max(axis=0).A1
			word_tfidf = dict(zip(feature_names, mean_tfidf))
			word_max_tfidf = dict(zip(feature_names, max_tfidf))

			logger.info(f"TF-IDF 벡터화 완료: {tfidf_matrix.shape}")

		except Exception as e:
			logger.error(f"TF-IDF 분석 실패: {e}")
			return None

		# 3) 고급 통계 계산
		doc_freq = {}
		word_positions = defaultdict(list)
		total_docs = len(texts)

		for doc_idx, text in enumerate(texts):
			words_in_doc = set(self.preprocessor.tokenize(text))
			for word in words_in_doc:
				if word not in doc_freq:
					doc_freq[word] = 0
				doc_freq[word] += 1
				word_positions[word].append(doc_idx)

		# 문서 빈도를 비율로 변환
		for word in doc_freq:
			doc_freq[word] = doc_freq[word] / total_docs

		# 4) 결과 데이터프레임 생성
		word_analysis = []
		for word in word_freq.keys():
			# 단어 특성 분석
			is_numeric = word.isdigit()
			is_alpha = word.isalpha()
			has_korean = bool(re.search(r'[가-힣]', word))
			has_english = bool(re.search(r'[a-zA-Z]', word))

			# 통계 계산
			freq = word_freq[word]
			doc_freq_ratio = doc_freq.get(word, 0)
			tfidf_mean = word_tfidf.get(word, 0)
			tfidf_max = word_max_tfidf.get(word, 0)

			# 집중도 계산 (특정 문서들에 집중되어 있는 정도)
			concentration = tfidf_max / (tfidf_mean + 1e-10) if tfidf_mean > 0 else 0

			word_analysis.append({
				'word': word,
				'frequency': freq,
				'doc_frequency': doc_freq_ratio,
				'tfidf_mean': tfidf_mean,
				'tfidf_max': tfidf_max,
				'concentration': concentration,
				'length': len(word),
				'is_numeric': is_numeric,
				'is_alpha': is_alpha,
				'has_korean': has_korean,
				'has_english': has_english,
				'column': column_name,
				'uniqueness_score': tfidf_mean * (1 - doc_freq_ratio)  # 독특함 점수
			})

		result_df = pd.DataFrame(word_analysis)
		result_df = result_df.sort_values('tfidf_mean', ascending=False)

		# 5) 메타데이터 저장
		self.metadata[column_name] = {
			'total_documents': total_docs,
			'total_words': len(all_words),
			'unique_words': len(word_freq),
			'avg_doc_length': np.mean(doc_word_counts),
			'vocabulary_density': len(word_freq) / len(all_words),
			'analysis_timestamp': pd.Timestamp.now()
		}

		# 6) 분석기와 결과 저장
		analyzer = DomainWordAnalyzer(column_name, self.config)
		analyzer.word_stats = result_df
		analyzer.metadata = self.metadata[column_name]

		self.analyzers[column_name] = analyzer
		self.results[column_name] = result_df

		return result_df

	def analyze_all_columns(self, table_name: str, text_columns: List[str],
	                        target_column: str = None, limit: int = 1000,
	                        sample_random: bool = False) -> Dict[str, pd.DataFrame]:
		"""모든 텍스트 컬럼에 대해 분석 수행"""

		# 데이터 로드
		column_data = self.load_text_columns(
			table_name, text_columns, target_column, limit, sample_random
		)

		if not column_data:
			logger.error("로드된 데이터가 없습니다.")
			return {}

		# 각 컬럼별 분석
		all_results = {}
		for col_name, data in column_data.items():
			if len(data['texts']) > 0:
				result = self.analyze_column(col_name, data['texts'], data['targets'])
				if result is not None:
					all_results[col_name] = result
			else:
				logger.warning(f"컬럼 {col_name}: 분석할 데이터가 없습니다.")

		return all_results

	def compare_columns(self, columns: List[str] = None) -> pd.DataFrame:
		"""컬럼별 분석 결과 비교 (개선된 버전)"""

		if columns is None:
			columns = list(self.results.keys())

		comparison_data = []

		for col in columns:
			if col in self.results:
				df = self.results[col]
				meta = self.metadata.get(col, {})

				# 각 컬럼별 통계
				stats = {
					'column': col,
					'total_documents': meta.get('total_documents', 0),
					'total_words': meta.get('total_words', 0),
					'unique_words': meta.get('unique_words', 0),
					'vocabulary_density': meta.get('vocabulary_density', 0),
					'avg_tfidf': df['tfidf_mean'].mean(),
					'max_tfidf': df['tfidf_mean'].max(),
					'avg_doc_freq': df['doc_frequency'].mean(),
					'high_tfidf_words': len(df[df['tfidf_mean'] > self.config.high_tfidf_threshold]),
					'stopword_candidates': len(df[
						                           (df['doc_frequency'] > self.config.stopword_doc_freq_threshold) &
						                           (df['tfidf_mean'] < self.config.stopword_tfidf_threshold)
						                           ]),
					'korean_words': len(df[df['has_korean'] == True]),
					'english_words': len(df[df['has_english'] == True]),
					'numeric_words': len(df[df['is_numeric'] == True])
				}
				comparison_data.append(stats)

		comparison_df = pd.DataFrame(comparison_data)

		# 시각화
		if len(comparison_df) > 0:
			self.visualize_column_comparison(comparison_df)

		return comparison_df

	def visualize_column_comparison(self, comparison_df: pd.DataFrame):
		"""컬럼별 비교 시각화 (개선된 버전)"""

		if len(comparison_df) == 0:
			logger.warning("시각화할 데이터가 없습니다.")
			return

		# 한글 폰트 설정
		plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'Arial Unicode MS']

		fig, axes = plt.subplots(2, 3, figsize=(18, 12))
		axes = axes.flatten()

		# 1) 총 단어 수 비교
		axes[0].bar(comparison_df['column'], comparison_df['unique_words'])
		axes[0].set_title('Unique Words by Column')
		axes[0].tick_params(axis='x', rotation=45)

		# 2) 어휘 밀도 비교
		axes[1].bar(comparison_df['column'], comparison_df['vocabulary_density'])
		axes[1].set_title('Vocabulary Density by Column')
		axes[1].tick_params(axis='x', rotation=45)

		# 3) 평균 TF-IDF 비교
		axes[2].bar(comparison_df['column'], comparison_df['avg_tfidf'])
		axes[2].set_title('Average TF-IDF by Column')
		axes[2].tick_params(axis='x', rotation=45)

		# 4) 고중요도 단어 수
		axes[3].bar(comparison_df['column'], comparison_df['high_tfidf_words'])
		axes[3].set_title(f'High TF-IDF Words (>{self.config.high_tfidf_threshold})')
		axes[3].tick_params(axis='x', rotation=45)

		# 5) 언어별 단어 분포
		x = np.arange(len(comparison_df))
		width = 0.25
		axes[4].bar(x - width, comparison_df['korean_words'], width, label='Korean', alpha=0.8)
		axes[4].bar(x, comparison_df['english_words'], width, label='English', alpha=0.8)
		axes[4].bar(x + width, comparison_df['numeric_words'], width, label='Numeric', alpha=0.8)
		axes[4].set_title('Word Types by Column')
		axes[4].set_xticks(x)
		axes[4].set_xticklabels(comparison_df['column'], rotation=45)
		axes[4].legend()

		# 6) 불용어 후보 수
		axes[5].bar(comparison_df['column'], comparison_df['stopword_candidates'])
		axes[5].set_title('Stopword Candidates')
		axes[5].tick_params(axis='x', rotation=45)

		plt.tight_layout()
		plt.show()

	def get_column_recommendations(self, column_name: str, top_n: int = 20) -> Optional[Dict]:
		"""특정 컬럼에 대한 추천사항 생성 (개선된 버전)"""

		if column_name not in self.results:
			logger.error(f"컬럼 {column_name}의 분석 결과가 없습니다.")
			return None

		df = self.results[column_name]

		# 불용어 후보 (높은 문서빈도 + 낮은 TF-IDF)
		stopwords = df[
			(df['doc_frequency'] > self.config.stopword_doc_freq_threshold) &
			(df['tfidf_mean'] < self.config.stopword_tfidf_threshold) &
			(df['length'] >= self.config.min_word_length)
			].head(top_n)['word'].tolist()

		# 중요 단어 후보 (높은 TF-IDF + 적절한 빈도)
		important = df[
			(df['tfidf_mean'] > 0.05) &
			(df['frequency'] >= 5) &
			(df['length'] >= self.config.min_word_length) &
			(df['is_alpha'] == True)
			].head(top_n)['word'].tolist()

		# 도메인 특화 단어 (높은 독특함 점수)
		domain_specific = df[
			(df['uniqueness_score'] > 0.01) &
			(df['length'] >= self.config.min_word_length)
			].nlargest(top_n, 'uniqueness_score')['word'].tolist()

		# 집중도가 높은 단어 (특정 문서에 집중)
		concentrated = df[
			(df['concentration'] > 2.0) &
			(df['tfidf_mean'] > 0.01)
			].nlargest(top_n, 'concentration')['word'].tolist()

		recommendations = {
			'column': column_name,
			'metadata': self.metadata.get(column_name, {}),
			'stopword_candidates': stopwords,
			'important_words': important,
			'domain_specific_words': domain_specific,
			'concentrated_words': concentrated,
			'top_tfidf_words': df.head(top_n)[
				['word', 'tfidf_mean', 'frequency', 'doc_frequency', 'uniqueness_score']
			].to_dict('records')
		}

		return recommendations

	def export_results(self, output_path: str = "analysis_results"):
		"""분석 결과를 파일로 내보내기"""

		output_path = Path(output_path)
		output_path.mkdir(exist_ok=True)

		# 각 컬럼별 결과 저장
		for col_name, df in self.results.items():
			df.to_csv(output_path / f"{col_name}_analysis.csv", index=False, encoding='utf-8-sig')

		# 메타데이터 저장
		with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
			# Timestamp를 문자열로 변환
			metadata_serializable = {}
			for col, meta in self.metadata.items():
				metadata_serializable[col] = {
					k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
					for k, v in meta.items()
				}
			json.dump(metadata_serializable, f, ensure_ascii=False, indent=2)

		# 컬럼 비교 결과 저장
		comparison_df = self.compare_columns()
		comparison_df.to_csv(output_path / "column_comparison.csv", index=False, encoding='utf-8-sig')

		logger.info(f"분석 결과가 {output_path}에 저장되었습니다.")


class DomainWordAnalyzer:
	"""도메인별 단어 분석기 (개선된 버전)"""

	def __init__(self, domain_name: str, config: Optional[AnalysisConfig] = None):
		self.domain_name = domain_name
		self.config = config or AnalysisConfig()
		self.word_stats = pd.DataFrame()
		self.metadata = {}

	def get_word_insights(self, word: str) -> Dict:
		"""특정 단어에 대한 상세 정보"""

		if self.word_stats.empty:
			return {}

		word_data = self.word_stats[self.word_stats['word'] == word]

		if word_data.empty:
			return {'word': word, 'found': False}

		word_info = word_data.iloc[0].to_dict()

		# 순위 정보 추가
		tfidf_rank = (self.word_stats['tfidf_mean'] > word_info['tfidf_mean']).sum() + 1
		freq_rank = (self.word_stats['frequency'] > word_info['frequency']).sum() + 1

		word_info.update({
			'found': True,
			'tfidf_rank': tfidf_rank,
			'frequency_rank': freq_rank,
			'total_words': len(self.word_stats)
		})

		return word_info


# 실제 사용 예시
def analyze_database_example():
	"""실제 데이터베이스 분석 예시"""

	# SQLite 연결 (실제 DB로 변경 필요)
	conn = sqlite3.connect('your_database.db')

	# 분석 설정
	config = AnalysisConfig(
		max_features=3000,
		min_df=3,
		high_tfidf_threshold=0.05
	)

	# 분석기 생성
	analyzer = DatabaseTextAnalyzer(conn, config)

	# 테이블 정보 확인
	table_info = analyzer.get_table_info('your_table')
	print("테이블 정보:", table_info)

	# 분석 실행
	results = analyzer.analyze_all_columns(
		table_name='your_table',
		text_columns=['title', 'content', 'description'],
		target_column='category',
		limit=5000,
		sample_random=True
	)

	# 컬럼별 비교
	comparison = analyzer.compare_columns()
	print("\n컬럼 비교 결과:")
	print(comparison)

	# 각 컬럼별 추천사항
	for col_name in results.keys():
		recommendations = analyzer.get_column_recommendations(col_name)
		print(f"\n=== {col_name} 컬럼 추천사항 ===")
		print(f"중요 단어: {recommendations['important_words'][:10]}")
		print(f"도메인 특화 단어: {recommendations['domain_specific_words'][:10]}")
		print(f"불용어 후보: {recommendations['stopword_candidates'][:10]}")

	# 결과 내보내기
	analyzer.export_results("./analysis_output")

	conn.close()
	return analyzer


if __name__ == "__main__":
	# 예시 실행
	# analyzer = analyze_database_example()
	pass
