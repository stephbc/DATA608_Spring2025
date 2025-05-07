import os
os.environ["RSTUDIO_CONNECT_APP_BASE_URL"] = "/"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

app_ui = ui.page_navbar(
    ui.nav_panel("Top 10",
        ui.card(
            ui.card_header("Most Purchased by Number of Orders"),
            ui.input_radio_buttons("rank_type", "", {"product": "Items", "aisle": "Departments"}),
            ui.output_plot("prod_bar"),
            full_screen=True,
            min_height=750,
        )
    ),
    ui.nav_panel("Customer Segmentation",
        ui.card(
            ui.card_header("Customer Segments Based on Order Frequency and Items Purchased"),
            ui.input_numeric("K", "Select Number of Clusters", 3),
            ui.output_plot("cluster"),
            full_screen=True,
            min_height=750,
        )
    ),
    title="Instacart Data Analysis"
)

def server(input: Inputs, output: Outputs, session: Session):
    
    @reactive.Calc
    def prods():
        return pd.read_csv("all_order_products.csv")

    @reactive.Calc
    def orders():
        return pd.read_csv("orders.csv")

    @reactive.Calc
    def items():
        return pd.read_csv("products.csv")

    @reactive.Calc
    def aisles():
        return pd.read_csv("aisles.csv")

    @reactive.Calc
    def df_tidy():
        df = prods().copy()
        df = df.drop(columns=["add_to_cart_order"])
        df["user_id"] = df["order_id"].map(orders().set_index("order_id")["user_id"])
        df["aisle_id"] = df["product_id"].map(items().set_index("product_id")["aisle_id"])
        return df

    @reactive.Calc
    def item_metrics():
        df = df_tidy()
        item_m = df.groupby(["product_id", "aisle_id"]).agg(
            num_orders=("order_id", "nunique")
        ).reset_index()
        item_m["product_name"] = item_m["product_id"].map(
            items().set_index("product_id")["product_name"]
        )
        item_m["aisle_name"] = item_m["aisle_id"].map(
            aisles().set_index("aisle_id")["aisle"]
        )
        return item_m

    @reactive.Calc
    def user_metrics():
        df = df_tidy()
        user_m = df.groupby("user_id").agg(
            num_orders=("order_id", "nunique"),
            num_items=("product_id", "count")
        ).reset_index()
        return user_m

    @reactive.Calc
    def top_all():
        items_df = item_metrics().sort_values("num_orders", ascending=False).head(10)
        items_df = items_df[["product_name", "num_orders"]].rename(columns={"num_orders": "product_orders"}).reset_index(drop=True)

        aisles_df = (
            item_metrics()
            .groupby(["aisle_id", "aisle_name"])
            .agg(aisle_orders=("num_orders", "sum"))
            .sort_values("aisle_orders", ascending=False)
            .head(10)
            .reset_index()
        )
        aisles_df = aisles_df[["aisle_name", "aisle_orders"]].reset_index(drop=True)

        return pd.concat([items_df, aisles_df], axis=1)

    @output
    @render.plot
    def prod_bar():
        plt.figure(figsize=(10, 6))
        df = top_all()
        x_col = "product_name" if input.rank_type() == "product" else "aisle_name"
        y_col = "product_orders" if input.rank_type() == "product" else "aisle_orders"
        sns.barplot(data=df, x=x_col, y=y_col, color="seagreen")
        plt.xlabel("")
        plt.ylabel("Number of Orders")
        plt.xticks(rotation=45)

    @output
    @render.plot
    def cluster():
        user_df = user_metrics()[["num_items", "num_orders"]].copy()
        X_scaled = StandardScaler().fit_transform(user_df)
        kmeans = KMeans(n_clusters=input.K(), random_state=101)
        user_df["Cluster"] = kmeans.fit_predict(X_scaled)

        sns.jointplot(
            data=user_df,
            x="num_items",
            y="num_orders",
            hue="Cluster",
            palette="colorblind",
            alpha=0.5
        )
        plt.xlabel("Total Number of Items")
        plt.ylabel("Number of Orders Placed")

app = App(app_ui, server)
