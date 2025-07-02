import pandas as pd

from common.init_psql import init_psql


def get_column_list(table_name: str, table_schema: str = 'public'):
	server, session = init_psql()
	engine = session.get_bind()

	# 2) 테이블의 컬럼 목록 읽기 (Postgres 정보스키마 활용)
	cols_df = pd.read_sql(
		f"""
		SELECT column_name
		FROM information_schema.columns
		WHERE table_schema = '{table_schema}'
		  AND table_name   = '{table_name}';
		""",
		con=engine
	)
	columns = cols_df['column_name'].tolist()

	server.stop()

	return columns


print(len(get_column_list('notice')))

# def count_categories(table_name: str, table_schema: str = 'public'):
