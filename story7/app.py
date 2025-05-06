# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shiny.express import input, render, ui
from shinywidgets import render_widget

# %%
# Load Data
prods = pd.read_csv('all_order_products.csv')
orders = pd.read_csv('orders.csv') # for the user_id
items = pd.read_csv('products.csv') # for the aisle_id
aisles = pd.read_csv('aisles.csv') # for aisle name

# %%
# Data Preprocessing
def df_tidy():
    df_tidy = prods.copy()
    df_tidy = df_tidy.drop(columns=['add_to_cart_order'])
    df_tidy['user_id'] = df_tidy['order_id'].map(orders.set_index('order_id')['user_id'])
    df_tidy['aisle_id'] = df_tidy['product_id'].map(items.set_index('product_id')['aisle_id'])
    return df_tidy

def item_metrics():
    item_m = df_tidy().groupby(['product_id', 'aisle_id']).agg({
        'order_id': 'nunique',
    }).rename(columns={
        'order_id': 'num_orders'
    }).reset_index()
    item_m['product_name'] = item_m['product_id'].map(items.set_index('product_id')['product_name'])
    item_m['aisle_name'] = item_m['aisle_id'].map(aisles.set_index('aisle_id')['aisle'])
    return item_m

def user_metrics():
    user_m = df_tidy().groupby('user_id').agg({
        'order_id': 'nunique',
        'product_id': 'count',
        # 'aisle_id': lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).rename(columns={
        'order_id': 'num_orders',
        'product_id': 'num_items'
        # 'aisle_id': 'top_cat_id'
    }).reset_index()
    # product_m['top_cat_name'] = product_m['top_cat_id'].map(aisles.set_index('aisle_id')['aisle'])
    return user_m

def top_all():
    top_items = item_metrics().sort_values(by = 'num_orders', ascending = False).head(10)
    top_items = top_items[['product_name', 'num_orders']].reset_index(drop = True)
    top_items = top_items.rename(columns={'num_orders': 'product_orders'})

    top_aisles = item_metrics().groupby(['aisle_id', 'aisle_name']).agg({'num_orders': 'sum'}).reset_index()
    top_aisles = top_aisles.sort_values(by = 'num_orders', ascending = False).head(10)
    top_aisles = top_aisles[['aisle_name', 'num_orders']].reset_index(drop = True)
    top_aisles = top_aisles.rename(columns={'num_orders': 'aisle_orders'})

    top_all = pd.merge(top_items, top_aisles, left_index = True, right_index = True)
    return top_all

# %%
# Shiny App
ui.page_opts(title = 'Instacart Data Analysis')

with ui.nav_panel('Top 10'):
    with ui.card(full_screen = True, min_height = 750):
        ui.card_header('Most Purchased by Number of Orders')
        ui.input_radio_buttons('rank_type', '', {'product': 'Items', 'aisle': 'Departments'})  

        @render.plot
        def prod_bar():
            plt.figure(figsize = (10, 6))
            sns.barplot(
                data = top_all(),
                palette = sns.light_palette('seagreen', reverse = True, n_colors = 20),
                x = f"{input.rank_type()}_name",
                y = f"{input.rank_type()}_orders")
            plt.xlabel('')
            plt.ylabel('Number of Orders')

with ui.nav_panel('Customer Segmentation'):
    with ui.card(full_screen = True, min_height = 750):
        ui.card_header('Customer Segments Based on Order Frequency and Items Purchased')
        ui.input_numeric('K', 'Select Number of Clusters', 3)

        @render.plot
        def cluster():
            user_df = user_metrics()[['num_items', 'num_orders']]
            X_scaled = StandardScaler().fit_transform(user_df)
            kmeans = KMeans(n_clusters = input.K(), random_state = 101)
            user_df['Cluster'] = kmeans.fit_predict(X_scaled)

            sns.jointplot(
                data = user_df,
                x = 'num_items',
                y = 'num_orders',
                hue = 'Cluster',
                palette = 'colorblind',
                alpha = 0.5)
            plt.xlabel('Total Number of Items')
            plt.ylabel('Number of Orders Placed')
            plt.legend(title = 'Cluster')
