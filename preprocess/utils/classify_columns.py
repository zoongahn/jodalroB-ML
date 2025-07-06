import pandas as pd
import os


def classify_columns(meta_df: pd.DataFrame):
    """
    Classify columns into numeric, categorical, text, and datetime based on metadata.

    - Numeric: 타입 in ['numeric', 'integer', 'bigint']
    - Categorical: 타입 in ['text', 'character(1)'] and 범주형 여부 == 'Y'
    - Text: 타입 in ['text', 'character(1)'] but 범주형 여부 != 'Y'
    - Datetime: 타입 == 'timestamp without time zone'
    """
    numeric = []
    categorical = []
    text_cols = []
    datetime_cols = []

    for _, row in meta_df.iterrows():
        col = row["컬럼명"]
        dtype = str(row["타입"]).lower()
        cat_flag = str(row.get("범주형 여부", "")).upper()

        if dtype in [
            "numeric",
            "integer",
            "bigint",
            "float",
            "double precision",
            "double",
            "real",
            "smallint",
            "smallserial",
            "serial",
            "bigserial",
            "money",
        ]:
            numeric.append(col)
        elif dtype in ["text", "character(1)", "character varying"]:
            if cat_flag == "Y":
                categorical.append(col)
            else:
                text_cols.append(col)
        elif dtype in [
            "timestamp without time zone",
            "timestamp with time zone",
            "date",
            "time without time zone",
            "time with time zone",
        ]:
            datetime_cols.append(col)
        else:
            # 기타 타입은 우선 텍스트로 분류
            text_cols.append(col)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "text": text_cols,
        "datetime": datetime_cols,
    }


def main():
    # 1) 메타 CSV 로드
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "meta", "notice.csv")
    meta_df = pd.read_csv(os.path.abspath(csv_path), dtype=str)

    # 2) 분류 수행
    column_groups = classify_columns(meta_df)

    # 3) 결과 출력
    for group, cols in column_groups.items():
        print(f"{group.capitalize()} columns ({len(cols)}): {cols}")


# 4) Datetime 처리 예시
# 실제 데이터프레임이 df_main이라 가정
# df_main = pd.read_csv('notice_full_data.csv')
# for dt_col in column_groups['datetime']:
#     df_main[dt_col] = pd.to_datetime(df_main[dt_col])
#     # 파생 변수 생성 예시
#     df_main[f"{dt_col}_year"]  = df_main[dt_col].dt.year
#     df_main[f"{dt_col}_month"] = df_main[dt_col].dt.month


if __name__ == "__main__":
    main()
