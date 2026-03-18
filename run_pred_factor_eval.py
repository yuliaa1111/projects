# python run_pred_factor_eval.py
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

from eval_all import complete_factor_analysis
from manage_db_read import ClickhouseReadOnly


BASE_DIR = Path("/home/quant/projects")

DB_SETTINGS_CANDIDATES = [
    BASE_DIR / "db_settings.json",
    Path("/home/quant/db_settings.json"),
]


def load_db_config() -> dict:
    for p in DB_SETTINGS_CANDIDATES:
        if p.exists():
            with open(p, "r") as f:
                settings = json.load(f)
            return settings["clickhouse_read"]
    raise FileNotFoundError("找不到 db_settings.json")


def connect_client() -> ClickhouseReadOnly:
    db_config = load_db_config()
    client = ClickhouseReadOnly(
        database=db_config["database"],
        host=db_config["host"],
        port=db_config["port"],
        username=db_config["username"],
        password=db_config["password"],
    )
    return client


def load_pred_as_factor_df(
    pred_dir: str,
    value_col: str = "y_pred",
    date_col: str = "date",
    stockid_col: str = "stockid",
    file_glob: str = "*.parquet",
) -> pd.DataFrame:
    pred_path = Path(pred_dir)
    files = sorted(pred_path.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {pred_dir}")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)

        need_cols = [date_col, stockid_col, value_col]
        miss = [c for c in need_cols if c not in df.columns]
        if miss:
            raise ValueError(f"{fp} missing columns: {miss}")

        tmp = df[need_cols].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        dfs.append(tmp)

    long_df = pd.concat(dfs, ignore_index=True)

    long_df = (
        long_df.sort_values([date_col, stockid_col])
        .drop_duplicates([date_col, stockid_col], keep="last")
        .reset_index(drop=True)
    )

    factor_df = (
        long_df.pivot(
            index=date_col,
            columns=stockid_col,
            values=value_col,
        )
        .sort_index()
    )

    return factor_df


if __name__ == "__main__":
    client = connect_client()

    pred_dir = "/home/quant/projects/runs/exp022_pca/sweeps/lgbm_04/preds"

    df = load_pred_as_factor_df(
        pred_dir=pred_dir,
        value_col="y_pred",
        date_col="date",
        stockid_col="stockid",
        file_glob="*_test.parquet",
    )

    print("factor df shape:", df.shape)
    print("date range:", df.index.min(), "->", df.index.max())
    print(df.iloc[:3, :3])

    complete_factor_analysis(
        factor_name="pred_lgbm_04ex_test",
        input_factor_df=df,
        start_date=datetime(2022, 1, 1, 0, 0, 0),
        end_date=datetime(2025, 12, 31, 0, 0, 0),
        frequency=1,
        factor_direction="positive",
        portfolio_type="long_only",
        long_groups=[9, 10],
        short_groups=[1, 2],
        group_num=10,
        neutralize=False,
        save_figures=True,
        save_dir="/home/quant/projects/runs/exp022_pca/factor_eval_lgbm_04ex_test",
        return_data=False,
        verbose=True,
        client=client,
        excess_return=True,
        eval_price="open",
    )