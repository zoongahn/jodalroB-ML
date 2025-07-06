import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psycopg2
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
import h5py
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# utils 모듈에서 classify_columns 함수 import
from preprocess.utils.classify_columns import classify_columns


class TextEmbeddingPipeline:
    """텍스트 컬럼 자동 Dense Embedding 파이프라인"""

    def __init__(self, db_config: Dict, model_name: str = "klue/roberta-base"):
        """
        Args:
            db_config: PostgreSQL 연결 정보
            model_name: 사용할 임베딩 모델명
        """
        self.db_config = db_config
        self.model_name = model_name
        self.engine = None
        self.model = None
        self.text_columns = []
        self.embeddings_dict = {}

        self._connect_db()
        self._load_model()
        self._identify_text_columns()

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
        """임베딩 모델 로드"""
        try:
            print(f"🔄 {self.model_name} 모델 로딩 중...")
            self.model = SentenceTransformer(self.model_name)
            print(
                f"✅ 모델 로딩 완료! 임베딩 차원: {self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise

    def _identify_text_columns(self):
        """메타데이터를 통해 텍스트 컬럼 식별"""
        try:
            # notice.csv 메타데이터 로드
            meta_path = project_root / "preprocess" / "meta" / "notice.csv"
            meta_df = pd.read_csv(meta_path, dtype=str)

            # 컬럼 분류
            column_groups = classify_columns(meta_df)
            self.text_columns = column_groups.get("text", [])

            print(f"📊 식별된 텍스트 컬럼 ({len(self.text_columns)}개):")
            for col in self.text_columns:
                print(f"  - {col}")

        except Exception as e:
            print(f"❌ 텍스트 컬럼 식별 실패: {e}")
            raise

    def load_data(self, limit: Optional[int] = None, sample_random: bool = True):
        """
        notice 테이블에서 텍스트 컬럼 데이터 로드

        Args:
            limit: 로드할 데이터 개수 제한
            sample_random: 랜덤 샘플링 여부
        """
        try:
            # 텍스트 컬럼들만 SELECT
            text_cols_str = ", ".join(self.text_columns)

            base_query = f"""
            SELECT {text_cols_str}
            FROM notice 
            WHERE 1=1
            """

            # 최소한 하나의 텍스트 컬럼이 NULL이 아닌 조건 추가
            conditions = []
            for col in self.text_columns:
                conditions.append(f"({col} IS NOT NULL AND TRIM({col}) != '')")

            if conditions:
                base_query += f" AND ({' OR '.join(conditions)})"

            # 랜덤 샘플링 또는 최신순
            if sample_random:
                base_query += " ORDER BY RANDOM()"
            else:
                base_query += " ORDER BY id DESC"

            # LIMIT 추가
            if limit:
                base_query += f" LIMIT {limit}"

            print(f"🔄 텍스트 데이터 로딩 중... (limit: {limit})")
            print(f"📄 쿼리: {base_query[:100]}...")

            self.df = pd.read_sql(base_query, self.engine)

            print(f"✅ 데이터 로딩 완료!")
            print(f"   - 로드된 데이터: {len(self.df)}개")
            print(f"   - 텍스트 컬럼: {len(self.text_columns)}개")

            # 각 컬럼별 NULL 비율 체크
            self._analyze_null_pattern()

        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")
            raise

    def _analyze_null_pattern(self):
        """각 텍스트 컬럼의 NULL 패턴 분석"""
        print("\n📊 텍스트 컬럼 NULL 분석:")
        print("-" * 60)

        for col in self.text_columns:
            if col in self.df.columns:
                total = len(self.df)
                null_count = self.df[col].isnull().sum()
                empty_count = (self.df[col] == "").sum()
                valid_count = total - null_count - empty_count

                print(
                    f"{col:20s}: "
                    f"유효 {valid_count:6d}개 ({valid_count/total*100:5.1f}%) | "
                    f"NULL {null_count:6d}개 | "
                    f"빈값 {empty_count:6d}개"
                )

    def _preprocess_text_column(
        self, column_data: pd.Series, column_name: str
    ) -> Tuple[List[str], List[int]]:
        """단일 텍스트 컬럼 전처리"""
        processed_texts = []
        valid_indices = []

        for i, text in enumerate(column_data):
            if pd.notna(text) and str(text).strip() != "":
                # 기본 전처리
                cleaned_text = str(text).strip()
                # 추가 전처리 (필요시)
                cleaned_text = self._clean_text(cleaned_text)

                if len(cleaned_text) >= 2:  # 최소 길이 체크
                    processed_texts.append(cleaned_text)
                    valid_indices.append(i)

        print(f"   {column_name}: {len(column_data)} → {len(processed_texts)}개 유효")
        return processed_texts, valid_indices

    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        import re

        # 특수문자 정리 (한글, 영어, 숫자, 기본 기호만 유지)
        text = re.sub(r"[^\w\s가-힣().-]", " ", text)
        # 연속된 공백 제거
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def create_embeddings(self, batch_size: int):
        """모든 텍스트 컬럼에 대해 임베딩 생성"""
        try:
            print(f"\n🔄 임베딩 생성 시작 (배치 크기: {batch_size})")
            print("=" * 80)

            for col in self.text_columns:
                if col not in self.df.columns:
                    print(f"⚠️  {col} 컬럼이 데이터프레임에 없음")
                    continue

                print(f"\n📝 {col} 컬럼 임베딩 생성 중...")

                # 전처리
                processed_texts, valid_indices = self._preprocess_text_column(
                    self.df[col], col
                )

                if len(processed_texts) == 0:
                    print(f"   ⚠️ {col}: 유효한 텍스트가 없음")
                    continue

                # 임베딩 생성
                embeddings = self.model.encode(
                    processed_texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )

                # 원본 크기로 복원 (NULL 위치는 0벡터)
                full_embeddings = np.zeros((len(self.df), embeddings.shape[1]))
                full_embeddings[valid_indices] = embeddings

                # 저장
                self.embeddings_dict[col] = {
                    "embeddings": full_embeddings,
                    "valid_indices": valid_indices,
                    "valid_count": len(valid_indices),
                    "embedding_dim": embeddings.shape[1],
                }

                print(
                    f"   ✅ {col}: {embeddings.shape[0]}개 임베딩 생성 완료 "
                    f"(차원: {embeddings.shape[1]})"
                )

            print(f"\n🎉 전체 임베딩 생성 완료!")
            self._print_embedding_summary()

        except Exception as e:
            print(f"❌ 임베딩 생성 실패: {e}")
            raise

    def _print_embedding_summary(self):
        """임베딩 생성 결과 요약"""
        print("\n📊 임베딩 생성 요약:")
        print("-" * 80)

        total_embeddings = 0
        for col, info in self.embeddings_dict.items():
            print(
                f"{col:20s}: "
                f"{info['valid_count']:6d}개 임베딩 | "
                f"차원: {info['embedding_dim']:4d} | "
                f"크기: {info['embeddings'].nbytes / (1024*1024):6.1f}MB"
            )
            total_embeddings += info["valid_count"]

        print(f"\n총 생성된 임베딩: {total_embeddings:,}개")

    def save_embeddings(self, output_dir: str = "embeddings_output"):
        """임베딩 결과 저장"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            print(f"\n💾 임베딩 저장 중... ({output_path})")

            # HDF5 형태로 압축 저장
            h5_path = (
                output_path
                / f"notice_embeddings_{self.model_name.replace('/', '_')}.h5"
            )

            with h5py.File(h5_path, "w") as f:
                # 메타데이터 저장
                f.attrs["model_name"] = self.model_name
                f.attrs["data_count"] = len(self.df)
                f.attrs["text_columns"] = [
                    col.encode("utf-8") for col in self.text_columns
                ]

                # 각 컬럼별 임베딩 저장
                for col, info in self.embeddings_dict.items():
                    group = f.create_group(col)
                    group.create_dataset(
                        "embeddings", data=info["embeddings"], compression="gzip"
                    )
                    group.create_dataset(
                        "valid_indices", data=info["valid_indices"], compression="gzip"
                    )
                    group.attrs["valid_count"] = info["valid_count"]
                    group.attrs["embedding_dim"] = info["embedding_dim"]

            # CSV로 메타데이터 저장
            meta_info = []
            for col, info in self.embeddings_dict.items():
                meta_info.append(
                    {
                        "column_name": col,
                        "valid_count": info["valid_count"],
                        "embedding_dim": info["embedding_dim"],
                        "file_size_mb": info["embeddings"].nbytes / (1024 * 1024),
                    }
                )

            meta_df = pd.DataFrame(meta_info)
            meta_df.to_csv(output_path / "embedding_metadata.csv", index=False)

            print(f"✅ 임베딩 저장 완료!")
            print(f"   - HDF5 파일: {h5_path}")
            print(f"   - 메타데이터: {output_path / 'embedding_metadata.csv'}")
            print(f"   - 총 파일 크기: {h5_path.stat().st_size / (1024*1024):.1f}MB")

        except Exception as e:
            print(f"❌ 임베딩 저장 실패: {e}")
            raise

    def load_saved_embeddings(self, h5_path: str) -> Dict:
        """저장된 임베딩 로드"""
        try:
            print(f"📂 임베딩 로드 중... ({h5_path})")

            embeddings_dict = {}

            with h5py.File(h5_path, "r") as f:
                model_name = f.attrs["model_name"]
                data_count = f.attrs["data_count"]

                print(f"   모델: {model_name}")
                print(f"   데이터 개수: {data_count:,}개")

                for col_name in f.keys():
                    group = f[col_name]
                    embeddings_dict[col_name] = {
                        "embeddings": group["embeddings"][:],
                        "valid_indices": group["valid_indices"][:],
                        "valid_count": group.attrs["valid_count"],
                        "embedding_dim": group.attrs["embedding_dim"],
                    }

                    print(f"   {col_name}: {group.attrs['valid_count']}개 임베딩")

            print("✅ 임베딩 로드 완료!")
            return embeddings_dict

        except Exception as e:
            print(f"❌ 임베딩 로드 실패: {e}")
            raise


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

    try:
        print("🚀 텍스트 컬럼 Dense Embedding 파이프라인 시작")
        print("=" * 80)

        # 파이프라인 초기화 (KoELECTRA 사용)
        pipeline = TextEmbeddingPipeline(
            db_config=db_config, model_name="monologg/koelectra-base-v3-discriminator"
        )

        # 데이터 로드 (테스트용 1000개)
        pipeline.load_data(limit=1000, sample_random=True)

        # 임베딩 생성
        pipeline.create_embeddings(batch_size=10)

        # 결과 저장
        pipeline.save_embeddings("notice_text_embeddings")

        print("\n🎉 모든 작업 완료!")

        # 사용 예시
        print("\n💡 임베딩 사용 예시:")
        print("# 특정 컬럼 임베딩 가져오기")
        print(
            "bidntcenm_embeddings = pipeline.embeddings_dict['bidntcenm']['embeddings']"
        )
        print("print(f'Shape: {bidntcenm_embeddings.shape}')")

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
