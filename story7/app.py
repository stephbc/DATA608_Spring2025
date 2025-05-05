import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shiny.express import input, render, ui
from shinywidgets import render_widget

from dfs import user_metrics, prod_metrics

ui.page_opts(title = "Instacart Data Analysis")

with ui.nav_panel("Table"):
    with ui.card():
        @render.data_frame
        def user_df():
            return render.DataTable(prod_metrics())

with ui.nav_panel("Customer Segmentation"):
    with ui.card(full_screen = True, min_height = 750):
        ui.card_header('Customer Segments Based on Order Frequency and Items Purchased')
        ui.input_numeric("K", "Select Number of Clusters", 3)

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
                alpha = 0.5)
            plt.xlabel('Total Number of Items')
            plt.ylabel('Number of Orders Placed')
            plt.legend(title = 'Cluster')