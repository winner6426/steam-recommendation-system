import argparse 
from pathlib import Path

import duckdb

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("--input", required = True)
parser.add_argument("--output-dir", required = True)
parser.add_argument("--limit", type = int, default = 100000)
args = parser.parse_args()

output = Path(args.output_dir)
output.mkdir(parents = True, exist_ok = True)

clean_path = output / "interactions_clean.parquet"
reject_path = output / "interaction_reject.parquet"

if clean_path.exists() or reject_path.exists():
    raise FileExistsError("Thư mục đã có đầu ra; hãy chọn thư mục mới.")

con = duckdb.connect()

con.execute("""
        CREATE TEMP TABLE raw AS 
        SELECT *
        FROM read_csv(?, header = true, all_varchar = true)
        LIMIT ? 
        """, [args.input, args.limit])

con.execute("""
    CREATE TEMP TABLE typed AS
    SELECT
        TRY_CAST(user_id AS BIGINT) AS user_id,
        TRY_CAST(app_id AS BIGINT) AS app_id,
        TRY_CAST(date AS DATE) AS date,
        CASE lower(trim(is_recommended))
            WHEN 'true' THEN true
            WHEN 'false' THEN false
            ELSE NULL
        END AS is_recommended,
        CASE
            WHEN isfinite(TRY_CAST(hours AS DOUBLE))
                 AND TRY_CAST(hours AS DOUBLE) >= 0
            THEN TRY_CAST(hours AS DOUBLE)
            ELSE NULL
        END AS hours,
        TRY_CAST(review_id AS BIGINT) AS review_id,
        user_id AS raw_user_id,
        app_id AS raw_app_id,
        date AS raw_date,
        is_recommended AS raw_label
    FROM raw
""")

con.execute("""
    CREATE TEMP VIEW checked AS
    SELECT *,
        (
            user_id IS NOT NULL AND user_id >= 0
            AND app_id IS NOT NULL AND app_id > 0
            AND date IS NOT NULL
            AND is_recommended IS NOT NULL
        ) AS valid
    FROM typed
""")

def sql_path(path): 
    return "'" + path.as_posix().replace("'", "''") + "'"

con.execute(f"""
    COPY (
        SELECT user_id, app_id, date, is_recommended, hours, review_id
        FROM checked 
        WHERE valid 
    ) TO {sql_path(clean_path)}
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

con.execute(f"""
    COPY (
        SELECT * FROM checked WHERE NOT valid
    ) TO {sql_path(reject_path)}
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print(con.execute("""
    SELECT
        count(*) AS input_rows,
        count(*) FILTER (WHERE valid) AS clean_rows,
        count(*) FILTER (WHERE NOT valid) AS rejected_rows,
        count(*) FILTER (
            WHERE valid AND is_recommended
        ) AS positive_rows,
        count(*) FILTER (
            WHERE valid AND NOT is_recommended
        ) AS negative_rows
    FROM checked
""").fetchdf())

print("Các cặp user-game bị lặp:")
print(con.execute("""
    SELECT user_id, app_id, count(*) AS occurrences
    FROM checked
    WHERE valid
    GROUP BY user_id, app_id
    HAVING count(*) > 1
    ORDER BY occurrences DESC
    LIMIT 10
""").fetchall())

written_rows = con.execute(
    "SELECT count(*) FROM read_parquet(?)",
    [clean_path.as_posix()],
).fetchone()[0]

expected_rows = con.execute(
    "SELECT count(*) FROM checked WHERE valid"
).fetchone()[0]

assert written_rows == expected_rows
print(f"Đã kiểm tra đọc lại Parquet: {written_rows} dòng.")
con.close()