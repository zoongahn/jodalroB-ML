import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import psycopg2
from sqlalchemy import create_engine
import re
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class NoticeClusteringPipeline:
    """공고제목 클러스터링 파이프라인"""

    def __init__(self, db_config: Dict):
        """
        Args:
            db_config: PostgreSQL 연결 정보
        """
        self.db_config = db_config
        self.engine = None
        self.model = None
        self.df = None
        self.embeddings = None
        self.cluster_labels = None
        self.cluster_results = None

        self._connect_db()
        self._load_model()

    def _connect_db(self):
        """PostgreSQL 연결"""
        try:
            conn_str = (
                f"postgresql://{self.db_config['user']}:"
                f"{self.db_config['password']}@"
                f"{self.db_config['host']}:"
                f"{self.db_config['port']}/"
                f"{self.db_config['database']}"
            )
            self.engine = create_engine(conn_str)
            print("✅ PostgreSQL 연결 성공!")
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            raise

    def _load_model(self):
        """KoELECTRA 모델 로드"""
        try:
            print("🔄 KoELECTRA 모델 로딩 중...")
            self.model = SentenceTransformer("monologg/koelectra-base-v3-discriminator")
            print("✅ KoELECTRA 모델 로딩 완료!")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise

    def load_data(self, limit: int = 10000, sample_random: bool = True):
        """
        공고제목 데이터 로드

        Args:
            limit: 가져올 데이터 개수
            sample_random: 랜덤 샘플링 여부
        """
        try:
            if sample_random:
                query = f"""
                SELECT bidntcenm 
                FROM notice 
                WHERE bidntcenm IS NOT NULL 
                AND TRIM(bidntcenm) != ''
                AND LENGTH(bidntcenm) >= 10
                ORDER BY RANDOM()
                LIMIT {limit}
                """
            else:
                query = f"""
                SELECT bidntcenm 
                FROM notice 
                WHERE bidntcenm IS NOT NULL 
                AND TRIM(bidntcenm) != ''
                AND LENGTH(bidntcenm) >= 10
                ORDER BY id DESC
                LIMIT {limit}
                """

            print(f"🔄 공고제목 데이터 로딩 중... (limit: {limit})")
            self.df = pd.read_sql(query, self.engine)

            # 중복 제거 및 전처리
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates(subset=["bidntcenm"])
            self.df["bidntcenm"] = self.df["bidntcenm"].apply(self._preprocess_title)
            self.df = self.df[
                self.df["bidntcenm"].str.len() >= 5
            ]  # 너무 짧은 제목 제거

            final_count = len(self.df)
            print(f"✅ 데이터 로딩 완료!")
            print(f"   - 초기: {initial_count}개")
            print(f"   - 중복제거 및 전처리 후: {final_count}개")
            print(f"   - 샘플 제목: {self.df['bidntcenm'].iloc[0]}")

        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")
            raise

    def _preprocess_title(self, title: str) -> str:
        """제목 전처리"""
        if pd.isna(title):
            return ""

        # 특수문자 정리 (한글, 영어, 숫자, 기본 기호만 유지)
        title = re.sub(r"[^\w\s가-힣().-]", " ", title)
        # 연속된 공백 제거
        title = re.sub(r"\s+", " ", title)
        return title.strip()

    def create_embeddings(self, batch_size: int = 32):
        """임베딩 생성"""
        try:
            print("🔄 임베딩 생성 중...")
            titles = self.df["bidntcenm"].tolist()

            self.embeddings = self.model.encode(
                titles,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

            print(f"✅ 임베딩 생성 완료! Shape: {self.embeddings.shape}")

        except Exception as e:
            print(f"❌ 임베딩 생성 실패: {e}")
            raise

    def perform_clustering(self, n_clusters: int = 50, random_state: int = 42):
        """K-means 클러스터링 수행"""
        try:
            print(f"🔄 클러스터링 수행 중... (n_clusters: {n_clusters})")

            # K-means 클러스터링
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300,
            )

            self.cluster_labels = kmeans.fit_predict(self.embeddings)

            # 클러스터 정보를 DataFrame에 추가
            self.df["cluster"] = self.cluster_labels

            print(f"✅ 클러스터링 완료!")
            print(f"   - 총 클러스터 수: {len(np.unique(self.cluster_labels))}")
            print(
                f"   - 가장 큰 클러스터 크기: {np.max(np.bincount(self.cluster_labels))}"
            )

        except Exception as e:
            print(f"❌ 클러스터링 실패: {e}")
            raise

    def analyze_clusters(self, top_k_keywords: int = 5, top_k_clusters: int = 20):
        """클러스터 분석 및 키워드 추출"""
        try:
            print("🔄 클러스터 분석 중...")

            results = []
            total_size = len(self.df)

            for cluster_id in range(len(np.unique(self.cluster_labels))):
                cluster_data = self.df[self.df["cluster"] == cluster_id]
                cluster_titles = cluster_data["bidntcenm"].tolist()
                cluster_size = len(cluster_titles)

                if cluster_size == 0:
                    continue

                # TF-IDF 키워드 추출
                tfidf_keywords = self._extract_tfidf_keywords(
                    cluster_titles, top_k_keywords
                )

                # 빈도 기반 키워드
                freq_keywords = self._extract_frequent_keywords(
                    cluster_titles, top_k_keywords
                )

                # 샘플 제목들
                samples = (
                    cluster_titles[:3] if len(cluster_titles) >= 3 else cluster_titles
                )

                results.append(
                    {
                        "cluster_id": cluster_id,
                        "size": cluster_size,
                        "percentage": (cluster_size / total_size) * 100,
                        "tfidf_keywords": tfidf_keywords,
                        "frequent_keywords": freq_keywords,
                        "samples": samples,
                        "avg_length": np.mean([len(title) for title in cluster_titles]),
                    }
                )

            # 크기순으로 정렬
            results.sort(key=lambda x: x["size"], reverse=True)
            self.cluster_results = results

            # 결과 출력
            self._print_cluster_analysis(results[:top_k_clusters])

        except Exception as e:
            print(f"❌ 클러스터 분석 실패: {e}")
            raise

    def _extract_tfidf_keywords(self, titles: List[str], top_k: int = 5):
        """TF-IDF 키워드 추출"""
        try:
            if len(titles) < 2:
                return []

            # 간단한 전처리
            processed_titles = [self._simple_preprocess(title) for title in titles]

            vectorizer = TfidfVectorizer(
                max_features=1000, ngram_range=(1, 2), min_df=2, max_df=0.95
            )

            tfidf_matrix = vectorizer.fit_transform(processed_titles)
            feature_names = vectorizer.get_feature_names_out()

            # 평균 TF-IDF 점수 계산
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # 상위 키워드
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]

            return keywords

        except Exception:
            return []

    def _extract_frequent_keywords(self, titles: List[str], top_k: int = 5):
        """빈도 기반 키워드 추출"""
        try:
            words = []
            for title in titles:
                # 간단한 단어 분리 (공백 기준)
                cleaned = re.sub(r"[^\w\s가-힣]", " ", title)
                words.extend([word for word in cleaned.split() if len(word) >= 2])

            # 빈도 계산
            word_counts = Counter(words)
            return word_counts.most_common(top_k)

        except Exception:
            return []

    def _simple_preprocess(self, text: str) -> str:
        """간단한 텍스트 전처리"""
        text = re.sub(r"[^\w\s가-힣]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _print_cluster_analysis(self, results: List[Dict]):
        """클러스터 분석 결과 출력"""
        print("\n" + "=" * 100)
        print("📊 클러스터 분석 결과")
        print("=" * 100)

        for i, cluster in enumerate(results):
            print(
                f"\n🔸 클러스터 {cluster['cluster_id']} (크기: {cluster['size']}개, {cluster['percentage']:.1f}%, 평균길이: {cluster['avg_length']:.1f}자)"
            )

            # TF-IDF 키워드
            if cluster["tfidf_keywords"]:
                tfidf_str = ", ".join(
                    [
                        f"{word}({score:.3f})"
                        for word, score in cluster["tfidf_keywords"]
                    ]
                )
                print(f"   🔑 TF-IDF: {tfidf_str}")

            # 빈도 키워드
            if cluster["frequent_keywords"]:
                freq_str = ", ".join(
                    [f"{word}({count})" for word, count in cluster["frequent_keywords"]]
                )
                print(f"   📈 빈도: {freq_str}")

            # 샘플 제목
            print(f"   📝 샘플:")
            for j, sample in enumerate(cluster["samples"], 1):
                print(f"      {j}. {sample}")

            if i >= 19:  # 상위 20개만 출력
                break

    def visualize_clusters(self, figsize: tuple = (15, 10)):
        """클러스터 시각화"""
        try:
            print("🔄 클러스터 시각화 중...")

            # PCA로 차원 축소
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(self.embeddings)

            # 시각화
            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # 1. 클러스터 산점도
            scatter = axes[0, 0].scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=self.cluster_labels,
                cmap="tab20",
                alpha=0.6,
                s=1,
            )
            axes[0, 0].set_title("클러스터 분포 (PCA)")
            axes[0, 0].set_xlabel("PC1")
            axes[0, 0].set_ylabel("PC2")

            # 2. 클러스터 크기 분포
            cluster_sizes = [result["size"] for result in self.cluster_results[:20]]
            cluster_ids = [
                f"C{result['cluster_id']}" for result in self.cluster_results[:20]
            ]

            axes[0, 1].bar(range(len(cluster_sizes)), cluster_sizes)
            axes[0, 1].set_title("상위 20개 클러스터 크기")
            axes[0, 1].set_xlabel("클러스터")
            axes[0, 1].set_ylabel("문서 수")
            axes[0, 1].tick_params(axis="x", rotation=45)

            # 3. 클러스터 크기 히스토그램
            all_sizes = [result["size"] for result in self.cluster_results]
            axes[1, 0].hist(all_sizes, bins=30, alpha=0.7)
            axes[1, 0].set_title("클러스터 크기 분포")
            axes[1, 0].set_xlabel("클러스터 크기")
            axes[1, 0].set_ylabel("빈도")

            # 4. 누적 커버리지
            cumulative_percentage = np.cumsum(
                [result["percentage"] for result in self.cluster_results]
            )
            axes[1, 1].plot(
                range(1, len(cumulative_percentage) + 1), cumulative_percentage
            )
            axes[1, 1].set_title("누적 커버리지")
            axes[1, 1].set_xlabel("클러스터 순위")
            axes[1, 1].set_ylabel("누적 비율 (%)")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            print("✅ 시각화 완료!")

        except Exception as e:
            print(f"❌ 시각화 실패: {e}")

    def save_results(self, output_path: str = "cluster_results.csv"):
        """결과 저장"""
        try:
            # 클러스터 정보가 포함된 DataFrame 저장
            self.df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"✅ 결과 저장 완료: {output_path}")

            # 클러스터 요약 정보 저장
            summary_path = output_path.replace(".csv", "_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("클러스터 분석 요약\n")
                f.write("=" * 50 + "\n\n")

                for result in self.cluster_results[:20]:
                    f.write(
                        f"클러스터 {result['cluster_id']} (크기: {result['size']}개)\n"
                    )
                    f.write(f"TF-IDF: {result['tfidf_keywords']}\n")
                    f.write(
                        f"샘플: {result['samples'][0] if result['samples'] else 'N/A'}\n"
                    )
                    f.write("-" * 30 + "\n")

            print(f"✅ 요약 저장 완료: {summary_path}")

        except Exception as e:
            print(f"❌ 저장 실패: {e}")


def main():
    """메인 실행 함수"""

    # 환경변수 로드
    load_dotenv()

    # PostgreSQL 설정
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "database": os.getenv("POSTGRES_DB"),
    }

    # 파이프라인 실행
    try:
        print("🚀 공고제목 클러스터링 파이프라인 시작")
        print("=" * 50)

        # 파이프라인 초기화
        pipeline = NoticeClusteringPipeline(db_config)

        # 데이터 로드 (10,000개로 시작)
        pipeline.load_data(limit=10000, sample_random=True)

        # 임베딩 생성
        pipeline.create_embeddings(batch_size=32)

        # 클러스터링 (50개 클러스터)
        pipeline.perform_clustering(n_clusters=50)

        # 클러스터 분석
        pipeline.analyze_clusters(top_k_keywords=5, top_k_clusters=20)

        # 시각화
        pipeline.visualize_clusters()

        # 결과 저장
        pipeline.save_results("notice_clustering_results.csv")

        print("\n🎉 클러스터링 완료!")

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
