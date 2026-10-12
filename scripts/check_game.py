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

con = duckdb.connect()

con.execute("""
    create temp table raw_game as 
    select * 
    from read_csv(?, header = true, all_varchar = true)
    limit ? 
    """, [args.input, args.limit])

con.execute("""
    create temp table gamevl as 
    select 
""")