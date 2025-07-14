import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import matplotlib.pyplot as plt
import requests
from typing import Literal, Union

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Getaround Rental Delay Dashboard",
    layout="wide",
    page_icon="🚗"
)


# Set global CSS for consistent font
st.markdown("""
<style>
    body {
        font-family: Arial, sans-serif;
    }
    .metric-title {
        font-family: Arial, sans-serif;
        font-size: 20px;
        font-weight: bold;
        color: #333333;
    }
    .row-height {
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .col-height {
        height: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ============================== UI HELPERS ==============================

def render_metric(label, value, bg_color, text_color):
    """
    Render a custom metric box.
    Accepts both numbers and strings (like percentages).
    """
    st.markdown(f"""
    <div class="metric-title">{label}</div>
    <div style='
        background-color:{bg_color};
        border-radius: 10%;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align:center;
        font-size:25px;
        font-weight: bold;
        width: 100px;
        height: 50px;
        margin: auto;
        color:{text_color};'>
        {value if isinstance(value, str) else f"{value:,}"}
    </div>
    """, unsafe_allow_html=True)

# ========================== LOAD AND CACHE DATA =========================
DATA_DELAY = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx"

@st.cache_data
def load_data(nrows=None):
    data_delay = pd.read_excel(DATA_DELAY)
    # Add a column to easily distinguish late returns from on-time returns (44% late)
    data_delay["return_status"] = "On time"
    data_delay.loc[data_delay["delay_at_checkout_in_minutes"] > 0, "return_status"] = "Late"
    # Add a column to define if the previous rental was late (nearly half of previous rentals were late)
    rental_status_map = data_delay.set_index('rental_id')['return_status'].to_dict()
    data_delay['previous_rental_status'] = data_delay['previous_ended_rental_id'].map(rental_status_map)
    # Add a column to know the delay of the previous rental
    delay_map = data_delay.set_index('rental_id')['delay_at_checkout_in_minutes'].to_dict()
    data_delay['previous_rental_delay'] = data_delay['previous_ended_rental_id'].map(delay_map)
    # Add a column to easily distinguish rentals followed by another rental (8.4%)
    data_delay['is_previous_rental'] = data_delay['rental_id'].isin(data_delay['previous_ended_rental_id'])
    # Add a column to easily distinguish rentals receded by another rental (8.6%)
    data_delay["has_previous_rental"] = "No"
    data_delay.loc[data_delay["previous_ended_rental_id"] > 0, "has_previous_rental"] = "Yes"
    # cleaning : trim + lowercase
    data_delay['checkin_type'] = data_delay['checkin_type'].str.strip().str.lower()
    data_delay['state'] = data_delay['state'].str.strip().str.lower()
    return data_delay

data_load_state = st.text('Loading data...')
data_delay = load_data(1000)
data_load_state.text("") # Change text from "Loading data..." to "" once the load_data function has run

#=================================== API =====================================

API_URL = "https://myriamgoyet-api-getaround.hf.space/predict"


def get_price_prediction(car_features: dict):
    response = requests.post(API_URL, json=car_features)
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        return prediction
    else:
        st.error(f"Erreur API: {response.status_code}")
        return None

# ================================ Side bar ==================================

page = st.sidebar.radio(
    "Navigation",
    options=["Rental Delay Analysis", "Price Prediction"],
    index=0,
    key="page_selection"
)

# Footer sidebar
st.sidebar.write("")
st.sidebar.write("Made with 💖 by [Myriam Goyet](https://github.com/myriamgoyet)")
st.sidebar.markdown("---")

# =========================== IMAGE, TITLE AND TEXT ==========================
st.markdown("""
    <div style>
        <h1><bold>Getaround Rental Delay Dashboard</bold></h1>
    </div>
""", unsafe_allow_html=True)

if page == "Rental Delay Analysis":
    image_path = 'https://images.tech.co/wp-content/uploads/2014/11/getaround.jpg'
    st.markdown(f'<img src="{image_path}" style="width:100%; height:auto;">', unsafe_allow_html=True)


#================================= PAGE 1 ======================================
#===============================================================================
# =========================== RENTAL OVERVIEW ==================================
if page == "Rental Delay Analysis":
    st.title("🚗 Rental Delay Analysis")
    st.divider()

    st.markdown("<div class='anchor' id='rental-overview'></div>", unsafe_allow_html=True)
    st.header("Rental Overview")

    col1, col2 = st.columns([1,1])

    with col1:
        # Create the first row with custom height
        row1 = st.container()
        row1.markdown('<div class="row-height">', unsafe_allow_html=True)
        render_metric("Number of car rental", f"{len(data_delay)}", "#40005B", "#ffffff")
        row1.markdown('</div>', unsafe_allow_html=True)

        # Create the second row with custom height
        row2 = st.container()
        row2.markdown('<div class="row-height">', unsafe_allow_html=True)
        render_metric("Number of unique cars rented", f"{data_delay['car_id'].nunique()}", "#40005B", "#ffffff")
        row2.markdown('</div>', unsafe_allow_html=True)
        

    with col2:
        checkin_counts = data_delay['checkin_type'].value_counts(normalize=True).reset_index()
        checkin_counts.columns = ['checkin_type', 'percentage']
        checkin_counts = checkin_counts.dropna()  # 👈 just in case

        # Create the pie chart
        fig = px.pie(
        checkin_counts,
        names='checkin_type',
        values='percentage',
        hole=0.3,
        color_discrete_sequence=["#40005B", "#AB63FA"]
        )

        # Update layout and formatting
        fig.update_traces(
            textinfo='percent',  # show only percentages inside slices
            textfont=dict(size=25, color='white'),    # increase font size 
            texttemplate='%{percent:.0%}',  # format with no decimals
            hovertemplate='%{label}: %{percent:.0%}',  # hover also without decimals
        )

        fig.update_layout(
            title_text="Checkin Type Distribution",
            title_font=dict(size=20, color="#333333", family="Arial"),
        )

        # Display
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ====================== PREVIOUS RENTAL =================================
    st.markdown("<div class='anchor' id='previous-rental'></div>", unsafe_allow_html=True)
    st.header("Previous Rental Overview")

    selected_type = st.selectbox(
            "Select previous rental type:",
            options=["All rentals", "Connect rentals", "Mobile rentals"]
        )

    # Apply filters
    if selected_type == "Connect rentals":
        filtered_df_prev = data_delay[data_delay['checkin_type'] == 'connect']
    elif selected_type == "Mobile rentals":
        filtered_df_prev = data_delay[data_delay['checkin_type'] == 'mobile']
    else:
        filtered_df_prev = data_delay  # All rentals


    col1, col2 = st.columns([1, 3])
    data_previous_rental = filtered_df_prev[filtered_df_prev["previous_ended_rental_id"].notna()].copy()

    with col1:
        # Create the first row with custom height
        row1 = st.container()
        row1.markdown('<div class="row-height">', unsafe_allow_html=True)
        render_metric("Cars with a previous rental",
                    f"{filtered_df_prev['previous_ended_rental_id'].notna().mean() * 100:.1f}%", 
                    "#40005B", "#ffffff")
        render_metric("Number of cars with a previous rental",
                    f"{len(filtered_df_prev[filtered_df_prev['previous_ended_rental_id'].notna()])}", 
                    "#40005B", "#ffffff")
        row1.markdown('</div>', unsafe_allow_html=True)

        # Create the second row with custom height
        row2 = st.container()
        row2.markdown('<div class="row-height">', unsafe_allow_html=True)
        render_metric("Delta median between 2 rentals (in min)",
                    f"{data_previous_rental['time_delta_with_previous_rental_in_minutes'].median()}", 
                    "#40005B", "#ffffff")
        row2.markdown('</div>', unsafe_allow_html=True)

        
    with col2:
        st.markdown(
            f"<div class='metric-title' id='delay-title'>Time delta with previous rental Distribution</div>",
            unsafe_allow_html=True
        )

        # Buckets de 30 min jusqu'à 720
        bucket_edges = list(range(0, 721, 30))  # 0, 30, 60, ..., 720
        bucket_labels = []

        for i in range(len(bucket_edges) - 1):
            start = bucket_edges[i]
            end = bucket_edges[i + 1] - 1
            if start == 0:
                bucket_labels.append("0 min")
            else:
                bucket_labels.append(f"{start}–{end} min")
        bucket_labels.append("Over 720 min")

        def categorize_delta(delta):
            if pd.isna(delta) or delta <= 0:
                return "0 min"
            elif delta > 720:
                return "Over 720 min"
            else:
                for start in range(30, 721, 30):  # buckets every 30 min
                    if delta <= start:
                        return f"{start - 30}–{start - 1} min"

        data_previous_rental['delta_bucket'] = data_previous_rental['time_delta_with_previous_rental_in_minutes'].apply(categorize_delta)

        delta_bucket_counts = (
            data_previous_rental['delta_bucket']
            .value_counts()
            .reindex(bucket_labels, fill_value=0)
            .reset_index()
        )
        delta_bucket_counts.columns = ['Delta range', 'Count']

        fig = px.bar(
            delta_bucket_counts,
            x='Delta range',
            y='Count',
            color='Delta range',
            color_discrete_sequence=["#40005B"]
        )

        fig.update_layout(
            xaxis_title="Time Delta Buckets",
            yaxis_title="Number of Rentals",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)


    st.divider()

    # =========================== DELAY ANALYSIS =================================
    st.markdown("<div class='anchor' id='delay-analysis'></div>", unsafe_allow_html=True)
    st.header("Delay Analysis")

    selected_type = st.selectbox(
            "Select rental type:",
            options=["All rentals", "Connect rentals", "Mobile rentals"]
        )
    # Second filter
    selected_previous_rental = st.selectbox(
        "Is the rental followed by an other rental reservation?",
        options=["All rentals", "Other rental reservation planed after check-out", "No other rental reservation planed after check-out"]
    )
    # Apply filters
    if selected_type == "Connect rentals":
        filtered_df_type = data_delay[data_delay['checkin_type'] == 'connect']
    elif selected_type == "Mobile rentals":
        filtered_df_type = data_delay[data_delay['checkin_type'] == 'mobile']
    else:
        filtered_df_type = data_delay  # All rentals

    if selected_previous_rental == "Other rental reservation planed after check-out":
        filtered_df_type = filtered_df_type[filtered_df_type['is_previous_rental'] == True]
    elif selected_previous_rental == "No other rental reservation planed after check-out":
        filtered_df_type = filtered_df_type[filtered_df_type['is_previous_rental'] == False]


    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-title' id='return-status-title'>Return Status Distribution</div>", unsafe_allow_html=True)
        return_counts = filtered_df_type['return_status'].value_counts(normalize=True).reset_index()
        return_counts.columns = ['Return status', 'percentage']
        return_counts['formatted'] = (return_counts['percentage'] * 100).round(1).astype(str) + '%'
        
        fig = px.bar(
        return_counts,
        x='Return status',
        y='percentage',
        text='formatted',
        color='Return status',
        color_discrete_sequence=["#005B0E", "#FA6363"]
        )

        fig.update_traces(
            textposition='outside',
            textfont=dict(size=14, color='black')
        )

        fig.update_layout(
            showlegend=False,  # ❌ no legend here
            yaxis_range=[0, 1],
            # ✅ Remove y-axis elements
            yaxis=dict(
                title='',          # remove y-axis title
                showticklabels=False,  # remove y-axis values (0%, 20%, ...)
                showgrid=False,    # remove horizontal grid lines
                zeroline=False     # remove line at y=0
            ),

            # ✅ Remove x-axis elements
            xaxis=dict(
                title='',          # remove x-axis title
                showgrid=False     # remove vertical grid lines (if any)
            ),

            legend_title_text=''
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            f"<div class='metric-title' id='delay-title'>Delay Distribution (≥ 1 min)</div>",
            unsafe_allow_html=True
        )

        def categorize_delay(delay):
            # Définir les intervalles de 30 minutes
            if delay <= 30:
                return "1–30 min"
            max_range = 600
            for i in range(60, max_range + 30, 30):
                lower_bound = i - 29
                upper_bound = i
                if delay <= i:
                    return f"{lower_bound}–{upper_bound} min"
            return f"Over {max_range} min"

        # Ne garder que les retards strictement positifs
        delay_only_df = filtered_df_type[filtered_df_type['delay_at_checkout_in_minutes'] > 0].copy()
        # Appliquer les catégories
        delay_only_df['delay_bucket'] = delay_only_df['delay_at_checkout_in_minutes'].apply(categorize_delay)

        # Générer les catégories pour le réindexage
        categories = ["1–30 min"] + [f"{i-29}–{i} min" for i in range(60, 600 + 30, 30)] + ["Over 600 min"]

        # Compter par bucket
        bucket_counts = (
            delay_only_df['delay_bucket']
            .value_counts()
            .reindex(categories, fill_value=0)
            .reset_index()
        )
        bucket_counts.columns = ['Delay range', 'Count']

        # Graphe
        fig = px.bar(
            bucket_counts,
            x='Delay range',
            y='Count',
            color='Delay range',
            color_discrete_sequence=["#40005B"]
        )
        fig.update_layout(
            xaxis_title="Delay Range",
            yaxis_title="Number of Rentals",
            showlegend=False,
            yaxis_range=[0, 3500]
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        render_metric("Median delay (in min)", f"{delay_only_df['delay_at_checkout_in_minutes'].median()}", "#40005B", "#ffffff")
    with col2:
        rentals_impacted_by_delay = filtered_df_type[
            (filtered_df_type["has_previous_rental"] == "Yes") &
            (filtered_df_type["previous_rental_status"] == "Late") &
            (filtered_df_type["previous_rental_delay"] > filtered_df_type["time_delta_with_previous_rental_in_minutes"])
        ]

        render_metric("Number of rentals impacted by delay", f"{rentals_impacted_by_delay.shape[0]}", "#40005B", "#ffffff")
        with st.expander("ℹ️ Where does this number come from?"):
                st.markdown("""
                A rental is impacted by delay if:
                - is preceded by an other rental 
                - AND the previous rental is delayed
                - AND the delay is superior to the time delta that separate both rentals
                In these conditions, the rental can not start on time.
            """)
    st.divider()
    # =========================== CANCELLATION OVERVIEW =================================
    st.markdown("<div class='anchor' id='cancellation-overview'></div>", unsafe_allow_html=True)
    st.header("Cancellation Overview")

    selected_type = st.selectbox(
            "Select delay type:",
            options=["All rentals", "Previous rental late", "Previous rental on time / no previous rental"]
        )

    # Apply filter
    if selected_type == "Previous rental late":
        filtered_df_delay = data_delay[data_delay['previous_rental_status'] == 'Late']
    elif selected_type == "Previous rental on time / no previous rental":
        filtered_df_delay = data_delay[(data_delay['previous_rental_status'] == 'On time') | (data_delay['previous_rental_status'].isna())]
    else:
        filtered_df_delay = data_delay  # All rentals



    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        state_counts = filtered_df_delay['state'].value_counts(normalize=True).reset_index()
        state_counts.columns = ['state', 'percentage']

        fig1 = px.pie(
            state_counts,
            names='state',
            values='percentage',
            hole=0.3,
            color_discrete_sequence=["#005B0E", "#FA6363"]
        )
        fig1.update_traces(
            textinfo='percent',
            textfont=dict(size=15, color='white'),
            texttemplate='%{percent:.0%}',
            hovertemplate='%{label}: %{percent:.0%}'
        )
        fig1.update_layout(
            title_text="All rentals",
            title_font=dict(size=20, color="#333333", family="Arial"),
            showlegend=False  # ❌ no legend here
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        render_metric("Number of rentals canceled", f"{filtered_df_delay[filtered_df_delay['state']=='canceled'].shape[0]}", "#40005B", "#ffffff")

    # ---------- Chart 2: Rentals with checkin_type = connect ----------
    with col2:
        connect_df = filtered_df_delay[filtered_df_delay['checkin_type'] == 'connect']
        connect_counts = connect_df['state'].value_counts(normalize=True).reset_index()
        connect_counts.columns = ['state', 'percentage']

        fig2 = px.pie(
            connect_counts,
            names='state',
            values='percentage',
            hole=0.3,
            color_discrete_sequence=["#005B0E", "#FA6363"]
        )
        fig2.update_traces(
            textinfo='percent',
            textfont=dict(size=15, color='white'),
            texttemplate='%{percent:.0%}',
            hovertemplate='%{label}: %{percent:.0%}'
        )
        fig2.update_layout(
            title_text="Connect rentals",
            title_font=dict(size=20, color="#333333", family="Arial"),
            showlegend=False  # ❌ no legend here
        )
        st.plotly_chart(fig2, use_container_width=True)

        render_metric("&nbsp;", f"{connect_df[connect_df['state']=='canceled'].shape[0]}", "#40005B", "#ffffff")

    # ---------- Chart 3: Rentals with checkin_type = mobile ----------
    with col3:
        mobile_df = filtered_df_delay[filtered_df_delay['checkin_type'] == 'mobile']
        mobile_counts = mobile_df['state'].value_counts(normalize=True).reset_index()
        mobile_counts.columns = ['state', 'percentage']

        fig3 = px.pie(
            mobile_counts,
            names='state',
            values='percentage',
            hole=0.3,
            color_discrete_sequence=["#005B0E", "#FA6363"]
        )
        fig3.update_traces(
            textinfo='percent',
            textfont=dict(size=15, color='white'),
            texttemplate='%{percent:.0%}',
            hovertemplate='%{label}: %{percent:.0%}'
        )
        fig3.update_layout(
            title_text="Mobile rentals",
            title_font=dict(size=20, color="#333333", family="Arial"),
            showlegend=True,  # ✅ legend only here
            legend=dict(
                orientation="h",         # horizontal legend
                yanchor="bottom",
                y=-0.3,                  # move legend below chart
                xanchor="center",
                x=0.5
        ))
        st.plotly_chart(fig3, use_container_width=True)
        
        render_metric("&nbsp", f"{mobile_df[mobile_df['state']=='canceled'].shape[0]}", "#40005B", "#ffffff")


    st.divider()

    # =========================== THRESHOLD AJUSTMENT =================================
    st.markdown("<div class='anchor' id='threshold-ajustment'></div>", unsafe_allow_html=True)
    st.header("Threshold Ajustment")

    # Create a slider
    min_value = 0
    max_value = 720
    default_value = 30  

    selected_value = st.slider(
        "Select a threshold :",
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        format="%d min"
    )

    filter_option = st.radio(
        "Choose filter option:",
        options=("Apply threshold to all rentals","Apply threshold to connect rentals only","Apply threshold to mobile rentals only" )
    )

    total_rentals = len(data_delay)
    total_connect = len(data_delay[data_delay['checkin_type'] == 'connect'])
    total_mobile = len(data_delay[data_delay['checkin_type'] == 'mobile'])
    rental_connect = data_delay[data_delay['checkin_type'] == 'connect']
    rental_mobile = data_delay[data_delay['checkin_type'] == 'mobile']

    threshold_filtered_rentals = data_delay[(data_delay['time_delta_with_previous_rental_in_minutes'] > selected_value) 
                                                |(data_delay['time_delta_with_previous_rental_in_minutes'].isna())
                                                |(data_delay['time_delta_with_previous_rental_in_minutes']<= 0)]

    threshold_filtered_connect = rental_connect[(rental_connect['time_delta_with_previous_rental_in_minutes'] > selected_value) 
                                                |(rental_connect['time_delta_with_previous_rental_in_minutes'].isna())
                                                |(rental_connect['time_delta_with_previous_rental_in_minutes']<= 0)]

    threshold_filtered_mobile = rental_mobile[(rental_mobile['time_delta_with_previous_rental_in_minutes'] > selected_value) 
                                                |(rental_mobile['time_delta_with_previous_rental_in_minutes'].isna())
                                                |(rental_mobile['time_delta_with_previous_rental_in_minutes']<= 0)]

    if filter_option == "Apply threshold to all rentals":
        percentage = (len(threshold_filtered_rentals) / total_rentals) * 100
        st.write(f"Percentage of rentals with threshold at {selected_value} min: {percentage:.2f}%")
        rentals_impacted_by_delay_with_threshold = data_delay[
            (data_delay["has_previous_rental"] == "Yes") &
            (data_delay["previous_rental_status"] == "Late") &
            (data_delay["previous_rental_delay"] > data_delay["time_delta_with_previous_rental_in_minutes"]) &
            (data_delay["previous_rental_delay"] > selected_value)
        ]
        st.write(f"Total number of rentals impacted by delay if threshold at {selected_value} min: {rentals_impacted_by_delay_with_threshold.shape[0]}")

        st.markdown(f"""
        - Connect rentals impacted by delay if threshold at {selected_value} min: {len(rentals_impacted_by_delay_with_threshold[rentals_impacted_by_delay_with_threshold['checkin_type'] == 'connect'])}
        - Mobile rentals impacted by delay if threshold at {selected_value} min: {len(rentals_impacted_by_delay_with_threshold[rentals_impacted_by_delay_with_threshold['checkin_type'] == 'mobile'])}
        """)

    elif filter_option == "Apply threshold to connect rentals only":
        percentage = ((len(threshold_filtered_connect) + len(rental_mobile)) / total_rentals) * 100
        st.write(f"Percentage of rentals with threshold at {selected_value} min for connect rentals only: {percentage:.2f}%")
        rentals_impacted_by_delay_mobile = data_delay[
            (data_delay["checkin_type"] == "mobile") &
            (data_delay["has_previous_rental"] == "Yes") &
            (data_delay["previous_rental_status"] == "Late") &
            (data_delay["previous_rental_delay"] > data_delay["time_delta_with_previous_rental_in_minutes"])
        ]
        connect_rentals_impacted_by_delay_with_threshold = data_delay[
            (data_delay["checkin_type"] == "connect") &
            (data_delay["has_previous_rental"] == "Yes") &
            (data_delay["previous_rental_status"] == "Late") &
            (data_delay["previous_rental_delay"] > data_delay["time_delta_with_previous_rental_in_minutes"]) &
            (data_delay["previous_rental_delay"] > selected_value)
        ]
        total_impacted = len(rentals_impacted_by_delay_mobile) + len(connect_rentals_impacted_by_delay_with_threshold)
        st.write(f"Total number of rentals impacted by delay if threshold at {selected_value} min for connect rentals only: {total_impacted}")

        st.markdown(f"""
        - Connect rentals impacted by delay if threshold at {selected_value} min: {len(connect_rentals_impacted_by_delay_with_threshold)}
        - Mobile rentals impacted by delay: {len(rentals_impacted_by_delay_mobile)}
        """)

    else:
        percentage = ((len(threshold_filtered_mobile) + len(rental_connect)) / total_rentals) * 100
        st.write(f"Percentage of rentals with threshold at {selected_value} min for mobile rentals only: {percentage:.2f}%")
        rentals_impacted_by_delay_connect = data_delay[
            (data_delay["checkin_type"] == "connect") &
            (data_delay["has_previous_rental"] == "Yes") &
            (data_delay["previous_rental_status"] == "Late") &
            (data_delay["previous_rental_delay"] > data_delay["time_delta_with_previous_rental_in_minutes"])
        ]
        mobile_rentals_impacted_by_delay_with_threshold = data_delay[
            (data_delay["checkin_type"] == "mobile") &
            (data_delay["has_previous_rental"] == "Yes") &
            (data_delay["previous_rental_status"] == "Late") &
            (data_delay["previous_rental_delay"] > data_delay["time_delta_with_previous_rental_in_minutes"]) &
            (data_delay["previous_rental_delay"] > selected_value)
        ]
        total_impacted = len(rentals_impacted_by_delay_connect) + len(mobile_rentals_impacted_by_delay_with_threshold)
        st.write(f"Total number of rentals impacted by delay if threshold at {selected_value} min for mobile rentals only: {total_impacted}")

        st.markdown(f"""
        - Connect rentals impacted by delay: {len(rentals_impacted_by_delay_connect)}
        - Mobile rentals impacted by delay if threshold at {selected_value} min: {len(mobile_rentals_impacted_by_delay_with_threshold)}
        """)


    st.divider()

    #================================= PAGE 2 ======================================
    #===============================================================================
elif page == "Price Prediction":
    st.title("💰 Price Prediction")

    car_type = st.selectbox("Car type", [""] + [
    'convertible', 'coupe', 'estate', 'hatchback', 'sedan',
    'subcompact', 'suv', 'van'
    ], index=0)
    model_key = st.selectbox("Brand / model", [""] + [
        'Audi', 'BMW', 'Citroën', 'Ferrari', 'Mercedes',
        'Mitsubishi', 'Nissan', 'Opel', 'PGO', 'Peugeot',
        'Renault', 'SEAT', 'Toyota', 'Volkswagen', 'other'
    ], index=0)
    mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=0, step=1000)
    engine_power = st.number_input("Engine power", min_value=0, max_value=1000, value=0, step=10)
    fuel = st.selectbox("Fuel", [""] + ['diesel', 'petrol'], index=0)
    has_gps = st.checkbox("GPS", value=False)
    automatic_car = st.checkbox("Automatic car", value=False)
    has_getaround_connect = st.checkbox("Has Getaround Connect", value=False)
    private_parking_available = st.checkbox("Has private parking available", value=False)
    has_speed_regulator = st.checkbox("Has speed regulator", value=False)
    has_air_conditioning = st.checkbox("Has A/C", value=False)
    winter_tires = st.checkbox("Has winter tires", value=False)
    paint_color = st.selectbox("Color", [""] +[
        'black', 'white', 'red', 'silver', 'grey', 'blue', 'beige', 'brown'
    ], index=0)

    car_features = {
        "car_type": car_type,
        "model_key": model_key,
        "mileage": mileage,
        "engine_power": engine_power,
        "fuel": fuel,
        "has_gps": has_gps,
        "automatic_car": automatic_car,
        "has_getaround_connect": has_getaround_connect,
        "private_parking_available": private_parking_available,
        "has_speed_regulator": has_speed_regulator,
        "has_air_conditioning": has_air_conditioning,
        "winter_tires": winter_tires,
        "paint_color": paint_color
    }

    if st.button("Predict Price"):
        if "" in [car_type, model_key, fuel, paint_color]:
            st.error("Please fill in all required fields before predicting.")
        elif mileage == 0 or engine_power == 0:
            st.error("Mileage and engine power must be greater than 0.")
        else:
            price = get_price_prediction(car_features)
            if price is not None:
                st.success(f"Predicted price: {price} €/Day")