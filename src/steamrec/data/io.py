"""Read source views; persist only validated tables."""
import json
from pathlib import Path
import duckdb

TABLES = ('interactions', 'games', 'users', 'content')

def load_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sql_path(path):
    return "'" + Path(path).as_posix().replace("'", "''") + "'"

def connect(config, directory, database=None):
    con = duckdb.connect(str(database) if database is not None else ':memory:')
    con.execute('SET memory_limit = ?', [config['execution']['memory_limit']])
    con.execute('SET threads = ?', [config['execution']['threads']])
    con.execute('SET temp_directory = ?', [str(directory)])
    con.execute('SET preserve_insertion_order = false')
    return con

def load_file(con, path, table_name):
    """Read all columns and rows via a view; never persist raw source tables."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if table_name not in TABLES:
        raise ValueError(f'Unknown source: {table_name}')
    if path.suffix.lower() == '.csv':
        reader = f'read_csv({sql_path(path)}, header=true, all_varchar=true)'
    elif path.suffix.lower() == '.json':
        reader = (f"read_json({sql_path(path)}, format='auto', columns="
                  "{app_id: 'VARCHAR', description: 'VARCHAR', tags: 'VARCHAR[]'})")
    else:
        raise ValueError(f'Unsupported format: {path.suffix}')
    con.execute(f'CREATE TEMP VIEW source_{table_name} AS SELECT * FROM {reader}')

def write_parquet(con, view, path):
    if view not in {f'{name}_{kind}' for name in TABLES for kind in ('ready', 'rejected')}:
        raise ValueError(f'Unknown output: {view}')
    if Path(path).exists():
        raise FileExistsError(path)
    con.execute(f'COPY (SELECT * FROM {view}) TO {sql_path(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')

def save_tables(con, staged):
    con.execute('BEGIN TRANSACTION')
    try:
        for name in TABLES:
            con.execute(f'CREATE TABLE {name} AS SELECT * FROM read_parquet({sql_path(staged[name])})')
        con.execute('COMMIT')
    except Exception:
        con.execute('ROLLBACK')
        raise
