"""Synthetic e-commerce warehouse (modelled on BigQuery's public thelook_ecommerce sample), 12 weeks of data.

Tables (dataset `shop`):
  users(user_id, country, signup_date, channel)
  products(product_id, name, category, brand, cost, retail_price)
  orders(order_id, user_id, created_at, status)
  order_items(item_id, order_id, product_id, quantity, sale_price)

Stored in a DuckDB file locally. load_to_bigquery() pushes the same tables to BigQuery (Parquet load jobs).
Seeded, so every run produces the same numbers.
"""
from __future__ import annotations

import datetime as dt
import os

import duckdb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "data", "warehouse.duckdb")
START = dt.date(2026, 8, 3)  # a Monday
WEEKS = 12
COUNTRIES = ["India", "United States", "United Kingdom", "Germany", "Brazil", "Japan", "Australia", "Canada"]
CHANNELS = ["Search", "Organic", "Social", "Email", "Display"]
CATEGORIES = {"Jeans": 55, "Tops & Tees": 25, "Outerwear": 120, "Sweaters": 70, "Shoes": 90, "Accessories": 20,
              "Activewear": 45, "Dresses": 65}
BRANDS = ["Northwind", "Kestrel", "Bluepeak", "Ivory", "Juniper", "Lumen", "Granite", "Harbor"]


def generate(path: str = DB_PATH, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    con = duckdb.connect(path)
    con.execute("CREATE SCHEMA shop")

    n_users = 6000
    country_p = np.array([0.24, 0.22, 0.12, 0.1, 0.1, 0.08, 0.07, 0.07])
    users = {
        "user_id": np.arange(1, n_users + 1),
        "country": rng.choice(COUNTRIES, n_users, p=country_p),
        "signup_date": [START + dt.timedelta(days=int(d)) for d in rng.integers(0, WEEKS * 7, n_users)],
        "channel": rng.choice(CHANNELS, n_users, p=[0.35, 0.25, 0.2, 0.12, 0.08]),
    }
    cats = list(CATEGORIES)
    n_products = 240
    p_cat = rng.choice(cats, n_products)
    p_price = np.array([round(CATEGORIES[c] * rng.uniform(0.6, 1.6), 2) for c in p_cat])
    products = {"product_id": np.arange(1, n_products + 1),
                "name": [f"{rng.choice(BRANDS)} {c} {i}" for i, c in enumerate(p_cat, 1)],
                "category": p_cat, "brand": rng.choice(BRANDS, n_products),
                "cost": np.round(p_price * rng.uniform(0.35, 0.6, n_products), 2), "retail_price": p_price}

    # Orders: volume grows ~2%/week, Outerwear and Sweaters pick up as autumn starts (a story for the report).
    rows_o, rows_i = [], []
    oid = iid = 0
    for day in range(WEEKS * 7):
        date = START + dt.timedelta(days=day)
        week = day // 7
        n = int(rng.poisson(95 * (1.02 ** week) * (1.15 if date.weekday() >= 5 else 1.0)))
        autumn = max(0.0, (week - 6) / 5)
        w = np.array([1.0 + (1.5 * autumn if c in ("Outerwear", "Sweaters") else 0)
                      - (0.4 * autumn if c == "Activewear" else 0) for c in p_cat])
        w = w / w.sum()
        for _ in range(n):
            oid += 1
            user = int(rng.integers(1, n_users + 1))
            status = rng.choice(["Complete", "Shipped", "Returned", "Cancelled"], p=[0.7, 0.15, 0.1, 0.05])
            rows_o.append((oid, user, date, str(status)))
            for _ in range(int(rng.integers(1, 4))):
                iid += 1
                pid = int(rng.choice(n_products, p=w)) + 1
                rows_i.append((iid, oid, pid, int(rng.integers(1, 3)),
                               round(float(p_price[pid - 1] * rng.uniform(0.8, 1.0)), 2)))

    con.execute("CREATE TABLE shop.users (user_id INTEGER, country VARCHAR, signup_date DATE, channel VARCHAR)")
    con.executemany("INSERT INTO shop.users VALUES (?,?,?,?)",
                    list(zip(users["user_id"].tolist(), users["country"].tolist(), users["signup_date"],
                             users["channel"].tolist())))
    con.execute("CREATE TABLE shop.products (product_id INTEGER, name VARCHAR, category VARCHAR, brand VARCHAR, "
                "cost DOUBLE, retail_price DOUBLE)")
    con.executemany("INSERT INTO shop.products VALUES (?,?,?,?,?,?)",
                    list(zip(products["product_id"].tolist(), products["name"], products["category"].tolist(),
                             products["brand"].tolist(), products["cost"].tolist(), products["retail_price"].tolist())))
    con.execute("CREATE TABLE shop.orders (order_id INTEGER, user_id INTEGER, created_at DATE, status VARCHAR)")
    con.executemany("INSERT INTO shop.orders VALUES (?,?,?,?)", rows_o)
    con.execute("CREATE TABLE shop.order_items (item_id INTEGER, order_id INTEGER, product_id INTEGER, "
                "quantity INTEGER, sale_price DOUBLE)")
    con.executemany("INSERT INTO shop.order_items VALUES (?,?,?,?,?)", rows_i)
    con.close()
    return path


def load_to_bigquery(project: str, dataset: str = "shop", path: str = DB_PATH):
    """Copy the four tables to BigQuery (needs google-cloud-bigquery and application-default credentials)."""
    import tempfile

    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    client.create_dataset(f"{project}.{dataset}", exists_ok=True)
    con = duckdb.connect(path, read_only=True)
    with tempfile.TemporaryDirectory() as tmp:
        for t in ("users", "products", "orders", "order_items"):
            pq = os.path.join(tmp, f"{t}.parquet")
            con.execute(f"COPY shop.{t} TO '{pq}' (FORMAT PARQUET)")
            cfg = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET,
                                         write_disposition="WRITE_TRUNCATE")
            with open(pq, "rb") as f:
                client.load_table_from_file(f, f"{project}.{dataset}.{t}", job_config=cfg).result()
    con.close()


if __name__ == "__main__":
    p = generate()
    con = duckdb.connect(p, read_only=True)
    for t in ("users", "products", "orders", "order_items"):
        print(t, con.execute(f"SELECT count(*) FROM shop.{t}").fetchone()[0])
