from sqlalchemy import text

from common.init_psql import init_psql
import pandas as pd
from tqdm import tqdm


def main():
	# 1) DB 연결
	server, session = init_psql()
	engine = session.get_bind()

	try:
		# 2) 테이블의 컬럼 목록 읽기 (Postgres 정보스키마 활용)
		cols_df = pd.read_sql(
			"""
			SELECT column_name
			FROM information_schema.columns
			WHERE table_schema = 'public'
			  AND table_name   = 'notice'
			  AND data_type  = 'text';
			""",
			con=engine
		)
		columns = cols_df['column_name'].tolist()

		# 3) 각 컬럼별 DISTINCT 카운트 계산
		results = []
		with tqdm(columns, desc="Counting categories") as pbar:
			for col in pbar:
				# update progress bar description with current column
				pbar.set_description(f"Counting {col}")
				sql = text(f"SELECT COUNT(DISTINCT {col}) AS num_categories FROM notice")
				cnt = session.execute(sql).scalar()
				results.append({'column': col, 'num_categories': cnt})

		# 4) DataFrame으로 변환 후 CSV로 저장
		out_df = pd.DataFrame(results)
		out_df.to_csv('notice_category_counts.csv', index=False, encoding='utf-8-sig')
		print("✅ notice_category_counts.csv 생성 완료")
	finally:
		session.close()
		if server:
			server.stop()


if __name__ == '__main__':
	main()
