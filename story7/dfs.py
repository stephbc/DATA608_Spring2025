from pathlib import Path

import pandas as pd

app_dir = Path(__file__).parent

prods = pd.read_csv(app_dir / "all_order_products.csv")
orders = pd.read_csv(app_dir / "orders.csv") # for the user_id
items = pd.read_csv(app_dir / "products.csv") # for the aisle_id
aisles = pd.read_csv(app_dir / "aisles.csv") # for aisle name

# %%
# Data Preprocessing
def df():
    df = prods.copy()
    df = df.drop(columns=['add_to_cart_order'])
    df['user_id'] = df['order_id'].map(orders.set_index('order_id')['user_id'])
    # df['aisle_id'] = df['product_id'].map(items.set_index('product_id')['aisle_id'])
    return df

# %%
# User Metrics
def user_metrics():
    user_m = df().groupby('user_id').agg({
        'order_id': 'nunique',
        'product_id': 'count',
        # 'aisle_id': lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).rename(columns={
        'order_id': 'num_orders',
        'product_id': 'num_items',
        # 'aisle_id': 'top_cat_id'
    }).reset_index()

    # product_m['top_cat_name'] = product_m['top_cat_id'].map(aisles.set_index('aisle_id')['aisle'])
    return user_m

# %%
# Product Metrics
def prod_metrics():
    product_m = df().groupby('product_id').agg({
        'order_id': 'nunique',
        'user_id': 'nunique',
    }).rename(columns={
        'order_id': 'num_orders',
        'user_id': 'num_users'
    }).reset_index()

    product_m['aisle_id'] = product_m['product_id'].map(items.set_index('product_id')['aisle_id'])
    product_m['aisle_name'] = product_m['aisle_id'].map(aisles.set_index('aisle_id')['aisle'])
    return product_m
