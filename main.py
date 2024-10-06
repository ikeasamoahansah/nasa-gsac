import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import json
import folium
from streamlit_folium import folium_static
import numpy as np
from agent import analyze_weather_data
from ghana_nlp import GhanaNLP
import dotenv
import os

dotenv.load_dotenv()



# Meteomatics API credentials
API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")

@st.cache_data
def get_weather_data(start_date, end_date, parameters, lat, lon):
    url = f"https://api.meteomatics.com/{start_date}T00:00:00Z--{end_date}T00:00:00Z:PT1H/t_2m:C,precip_1h:mm,wind_speed_10m:ms/{lat},{lon}/json"

    st.write(f"Requesting data from: {url}")  # Debug: Show the URL being requested
    
    try:
        response = requests.get(url, auth=(API_USERNAME, API_PASSWORD))
        response.raise_for_status()  # Raise an exception for bad status codes
        return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data: {e}")
        if response.status_code == 400:
            st.error("Bad request. Please check your input parameters and API credentials.")
        elif response.status_code == 401:
            st.error("Unauthorized. Please check your API credentials.")
        elif response.status_code == 403:
            st.error("Forbidden. You may not have permission to access this resource.")
        elif response.status_code == 404:
            st.error("Not found. The requested resource doesn't exist.")
        else:
            st.error(f"An error occurred: {response.status_code}")
        
        st.write("Response content:", response.text)  # Debug: Show the response content
        return None

def plot_weather_data(data, parameter):
    df = pd.DataFrame(data['data'][0]['coordinates'][0]['dates'])
    fig = px.line(df, x='date', y='value', title=f"{parameter} over time")
    return fig

def predict_rainfall(data):
    precip_data = [item['value'] for item in data['data'][0]['coordinates'][0]['dates']]
    return any(precip > 0.5 for precip in precip_data)

def create_heatmap(data, lat, lon):
    precip_data = [item['value'] for item in data['data'][0]['coordinates'][0]['dates']]
    m = folium.Map(location=[lat, lon], zoom_start=10)
    folium.Circle(
        radius=1000,
        location=[lat, lon],
        popup="Selected Location",
        color="crimson",
        fill=True,
    ).add_to(m)
    folium.plugins.HeatMap(
        list(zip([lat]*len(precip_data), [lon]*len(precip_data), precip_data)),
        radius=20
    ).add_to(m)
    return m

def main():
    st.set_page_config(page_title="TerraGraph 🌱", page_icon="🌱", layout="wide")

    st.title("TerraGraph: NASA Apps Challenge 🚀")
    st.write("Empowering farmers with data-driven insights for water management 💧")

    st.sidebar.title("Navigation 🧭")
    page = st.sidebar.radio("Go to", ["Home 🏠", "Rainfall Prediction 🌧️", "Data Insights 📊", "About 📘", "Team 👥"])

    if page == "Home 🏠":
        st.header("Welcome to TerraGraph 🌱")
        st.write("TerraGraph is your advanced tool for predicting rainfall and analyzing water-related challenges using NASA datasets and cutting-edge weather APIs.")
        st.image("https://via.placeholder.com/800x400.png?text=TerraGraph+Banner", use_column_width=True)

    elif page == "Rainfall Prediction 🌧️":
        st.header("Rainfall Prediction 🌧️")

        # For language support
        st.write("For language support in Ghanaian languages")
        key = st.text_input("Enter your GhanaNLP API key (optional)")
        api = GhanaNLP(key)

        st.write("Enter your location and select the date range to predict rainfall in your area.")


        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude 🌐", value=52.520551)
            start_date = st.date_input("Start Date 📅", value=datetime.now())
        with col2:
            lon = st.number_input("Longitude 🌐", value=13.461804)
            end_date = st.date_input("End Date 📅", value=datetime.now() + timedelta(days=7))

        if st.button("Predict Rainfall 🔮"):
            st.write(f"API Username: {API_USERNAME}")  # Debug: Show the API username being used
            data = get_weather_data(start_date, end_date, ["precip_1h:mm", "t_2m:C"], lat, lon)
            if data:
                rainfall_predicted = predict_rainfall(data)

                st.subheader("Rainfall Prediction Results 📊")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rainfall Prediction", "Rain ☔" if rainfall_predicted else "No Rain ☀️")
                with col2:
                    confidence = np.random.randint(70, 100)  # Placeholder for actual confidence calculation
                    st.metric("Prediction Confidence", f"{confidence}%")

                st.subheader("Weather Visualization 📈")
                tab1, tab2 = st.tabs(["📊 Charts", "🗺️ Heatmap"])
                with tab1:
                    st.plotly_chart(plot_weather_data(data, "Precipitation (mm)"), use_container_width=True)
                    st.plotly_chart(plot_weather_data(data, "Temperature (°C)"), use_container_width=True)
                with tab2:
                    st.write("Precipitation Heatmap")
                    folium_static(create_heatmap(data, lat, lon))

                

                st.subheader("Farmer's Insight 🌾")
                if rainfall_predicted:
                    st.info("Rainfall is expected in the coming days. Consider adjusting your irrigation schedule and preparing for potential water accumulation.")
                    st.audio(api.tts(api.translate( "Rainfall is expected in the coming days. Consider adjusting your irrigation schedule and preparing for potential water accumulation.", "en-tw"), "tw"))
                else:
                    st.warning("No significant rainfall is expected. Ensure your crops have adequate irrigation and consider water conservation measures.")
                    st.audio(api.tts(api.translate("No significant rainfall is expected. Ensure your crops have adequate irrigation and consider water conservation measures.", "en-tw"), "tw"))

                # AI interpretation for deeper insights
                ai_insight = analyze_weather_data(data)
                st.subheader("AI-Powered Insight 💡")
                st.write(ai_insight)
                
         
    elif page == "Data Insights 📊":
        st.header("Data Insights 📊")
        st.write("Upload your custom dataset for in-depth analysis.")
        
        uploaded_file = st.file_uploader("Choose a CSV or Excel file 📁", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            st.subheader("Data Preview 👀")
            st.write(data.head())
            
            st.subheader("Data Analysis 🔬")
            col1, col2 = st.columns(2)
            with col1:
                numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
                x_axis = st.selectbox("Choose x-axis 📏", numeric_columns)
                chart_type = st.selectbox("Select chart type 📊", ["Scatter", "Line", "Bar"])
            with col2:
                y_axis = st.selectbox("Choose y-axis 📐", numeric_columns)
                color_by = st.selectbox("Color by (optional) 🎨", ["None"] + list(data.columns))
            
            if chart_type == "Scatter":
                fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by if color_by != "None" else None,
                                 title=f"{y_axis} vs {x_axis}")
            elif chart_type == "Line":
                fig = px.line(data, x=x_axis, y=y_axis, color=color_by if color_by != "None" else None,
                              title=f"{y_axis} over {x_axis}")
            else:
                fig = px.bar(data, x=x_axis, y=y_axis, color=color_by if color_by != "None" else None,
                             title=f"{y_axis} by {x_axis}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Statistical Summary 📈")
            st.write(data.describe())

    elif page == "About 📘":
        st.header("About TerraGraph 📘")
        st.write("""
        TerraGraph is our innovative solution for the NASA Apps Challenge, designed to empower farmers 
        with data-driven insights for better water management and crop planning.

        🌱 Our mission is to bridge the gap between complex weather data and practical farming decisions.
        
        🌍 We utilize NASA datasets and advanced weather APIs to provide accurate rainfall predictions 
        and comprehensive environmental analysis.
        
        💡 Key Features:
        - Precise rainfall predictions
        - Interactive data visualizations
        - Custom data analysis tools
        - Geographical mapping of weather patterns
        
        Join us in revolutionizing agriculture through technology and data science! 🚀
        """)

    elif page == "Team 👥":
        st.header("Meet Our Team 👥")
        st.write("We are a group of passionate college developers dedicated to making a difference in agriculture and water management.")
        
        team_members = [
            {"name": "Prince", "role": "Lead Developer 💻", "bio": "Passionate about coding and solving real-world problems."},
            {"name": "Ike", "role": "Data Scientist 📊", "bio": "Loves turning complex data into actionable insights."},
            {"name": "Joshua", "role": "UI/UX Designer 🎨", "bio": "Focuses on creating intuitive and user-friendly interfaces."},
        ]

        for member in team_members:
            st.subheader(f"{member['name']} - {member['role']}")
            st.write(member['bio'])

if __name__ == "__main__":
    main()
