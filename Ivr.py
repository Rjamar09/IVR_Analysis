import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# Load your image
image = Image.open('D:\AOSPL\Automation\ivr_project\logo.jpg')

# Center-align the title and change the color using HTML-style formatting
title_html = """
    <style>
        .title {
            color: skyblue; 
            text-align: center;
        }
    </style>
    <h1 class="title">IVR Data Analysis Dashboard</h1>
"""
st.markdown(title_html, unsafe_allow_html=True)

# Display the image
st.image(image, use_column_width=True)

# Function to clean phone numbers in a DataFrame
def clean_phone_numbers(df, column_name):
    df[column_name] = df[column_name].astype(str)  # Convert the column to string
    df[column_name] = df[column_name].str.replace(',', '')  # Remove commas from phone numbers

# Main Streamlit code
st.title("IVR Data Analysis")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file, delimiter=';')  # Specify the delimiter if needed
    else:
        df = pd.read_excel(uploaded_file)

    # Sidebar information
    st.sidebar.subheader("File Information")
    st.sidebar.write(f"Uploaded File: {uploaded_file.name}")
    st.sidebar.write(f"Number of Rows: {df.shape[0]}")
    st.sidebar.write(f"Number of Columns: {df.shape[1]}")

    # Data cleaning (if phone numbers contain commas)
    phone_column_name = 'phone_number'  # Replace with the actual column name
    if phone_column_name in df.columns:
        clean_phone_numbers(df, phone_column_name)

    # Data analysis section
    st.subheader("IVR Data Overview")
    st.dataframe(df)
    
    # Data Cleaning Process
    # Calculate missing value percentages
    missing_percentages = df.isnull().sum()/len(df)*100

    # Identify columns with the most missing values
    max_missing_columns = df.columns[missing_percentages == missing_percentages.max()]

    # Identify columns to drop
    columns_to_drop = []
    for col in df.columns:
        if df[col].isnull().all():
         columns_to_drop.append(col)
    columns_to_drop.append('security_phrase')

    # Drop columns with all missing values and 'security_phrase'
    data = df.drop(columns=columns_to_drop, axis=1)

    columns_to_drop = ['status', 'list_description', 'gmt_offset_now','date_of_birth','phone_code','alt_dial','rank']
    data= data.drop(columns=columns_to_drop) # Drop multiple columns by names
    
    # Create a new column for date
    data['date'] = data['call_date'].dt.date
    # Create a new column for time
    data['time'] = data['call_date'].dt.time

    # Proper formatting
    data = data[['call_date', 'date', 'time', 'phone_number', 'user', 'full_name', 'campaign_id', 'list_id', 'first_name', 'address1', 'length_in_sec', 'lead_id', 'list_name', 'status_name']]

    # Call Volume Analysis
    # Assuming you have your DataFrame loaded into 'data'
    data['date'] = pd.to_datetime(data['date'])
    daily_call_volume = data.groupby(data['date']).size().reset_index(name='call_count')

    # Plot daily call volume trends using Plotly
    st.subheader('Daily Call Volume Trend')

    # Create a bar plot using Plotly Express with different colors
    fig = px.bar(daily_call_volume, x='date', y='call_count', color='call_count', color_continuous_scale='Viridis')

    # Customize x-axis tick labels
    fig.update_xaxes(tickformat='%Y-%m-%d', tickangle=45)

    # Set the spacing between bars and annotations
    spacing = 50  # Adjust this value for desired spacing

    # Add annotations above the bars with spacing
    annotations = []
    for idx, row in daily_call_volume.iterrows():
        annotations.append(
            dict(
                x=row['date'],
                y=row['call_count'] + spacing,  # Add spacing to y position
                text=str(row['call_count']),
                showarrow=False,
                font=dict(color='orange')  # Set annotation text color to orange
            )
        )

    fig.update_layout(annotations=annotations)

    # Display the Plotly figure using Streamlit
    st.plotly_chart(fig)


    # Get the value counts of 'company_name'
    user_counts = df['user'].value_counts()

    # Create a bar plot using Plotly Express
    fig = px.bar(user_counts, x=user_counts.index, y=user_counts.values, color=user_counts.index)

    # Update the layout of the plot
    fig.update_layout(
        title='User Counts',
        xaxis_title='Company Name',
        yaxis_title='Count',
        xaxis=dict(tickangle=-45),
        showlegend=False
    )
    
    # Set the spacing between bars and annotations
    spacing = 50  # Adjust this value for desired spacing
    
    # Add annotations to the bars
    annotations = []
    for x, y in zip(user_counts.index, user_counts.values):
        annotations.append(
            dict(
                x=x,
                y=y + spacing,
                text=str(y),
                showarrow=False,
                font=dict(color='orange')
            )
        )
    fig.update_layout(annotations=annotations)

    # Streamlit app
    st.title('User Counts Bar Plot')
    st.plotly_chart(fig, use_container_width=True)



    # Call Volume Analysis
    data['date'] = pd.to_datetime(data['date'])
    daily_call_volume = data.groupby(data['date'].dt.to_period('D')).size()
    weekly_call_volume = data.groupby(data['date'].dt.to_period('W')).size()
    monthly_call_volume = data.groupby(data['date'].dt.to_period('M')).size()

    # Convert period objects to strings
    daily_call_volume.index = daily_call_volume.index.astype(str)
    weekly_call_volume.index = weekly_call_volume.index.astype(str)
    monthly_call_volume.index = monthly_call_volume.index.astype(str)

    # Convert the 'date' column to datetime format
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S')

    #Identify peak calling hours
    hourly_call_volume = data.groupby(data['time'].apply(lambda x: x.hour)).size()

    # Plot hourly call volume 
    st.subheader('Hourly Call Volume Trends')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=hourly_call_volume.index, y=hourly_call_volume.values)
    plt.xlabel('Hour')
    plt.ylabel('Call Volume')
    plt.title('Hourly Call Volume Trends')

    # Add annotations to the bars
    for index, value in enumerate(hourly_call_volume.values):
        plt.text(index, value, str(value), ha='center', va='bottom')
    st.pyplot(plt)

   # Convert 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Calculate daily call volume
    daily_call_volume = data.groupby(data['date'].dt.day_name()).size()

    # Reorder the days of the week for proper sequence
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_call_volume = daily_call_volume.reindex(ordered_days)

    # Streamlit app
    st.subheader('Peak Calling Days Analysis')
    # Display the bar plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=daily_call_volume.index, y=daily_call_volume.values)
    plt.xlabel('Day')
    plt.ylabel('Call Volume')
    plt.title('Peak Calling Days')

    # Add annotations to the bars
    for index, value in enumerate(daily_call_volume.values):
        plt.text(index, value, str(value), ha='center', va='bottom')

    # Display the plot using st.pyplot()
    st.pyplot(plt)

    # List of unique days
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Allow user to select specific days
    selected_days = st.multiselect('Select days to view data', ordered_days)
    # Display call data for selected days
    if selected_days:
        selected_data = data[data['call_date'].dt.day_name().isin(selected_days)]     
        # Get unique dates from the selected data
        unique_dates = selected_data['call_date'].dt.date.unique()        
        # Allow user to select a specific date from the filtered data
        selected_date = st.selectbox('Select a date from filtered data', unique_dates)        
        # Filter data for the selected date
        selected_date_data = selected_data[selected_data['call_date'].dt.date == selected_date]      
        # Get unique companies for the selected date
        unique_companies_for_date = selected_date_data['user'].unique()
        # Allow user to select a company for the selected date
        selected_company = st.selectbox('Select a company for the selected date', unique_companies_for_date)

        # Display call data for selected date and company
        if selected_company:
            selected_company_data = selected_date_data[selected_date_data['user'] == selected_company]
            selected_company_data = selected_company_data[['call_date', 'user', 'phone_number', 'first_name', 'length_in_sec', 'status_name']]

            # Add index number to the dataframes
            selected_date_data = selected_date_data.reset_index(drop=True)
            selected_company_data = selected_company_data.reset_index(drop=True)

            # Display the filtered data in a table
            st.write('Call data for selected date and company:')
            st.dataframe(selected_company_data, width=800)
            
     # Streamlit app code
    st.subheader('Status Analysis')

    # Group the result by status_name and count the occurrences
    grouped_result = data['status_name'].value_counts()

    # Create a bar plot using Seaborn   
    plt.figure(figsize=(10, 6))
    sns.barplot(x=grouped_result.index, y=grouped_result.values)
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.title('Count of Occurrences by Status')
    plt.xticks(rotation=45, ha='right')

    # Add annotations to each bar
    for i, value in enumerate(grouped_result.values):
        plt.text(i, value, str(value), ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Group the data by 'status_name' and calculate the count of occurrences
    status_counts = data['status_name'].value_counts()

    # Create a pie chart using Plotly Express
    fig = px.pie(status_counts, names=status_counts.index, values=status_counts.values,
                 title='Distribution of Call Status')

    # Customize the title font size
    fig.update_layout(title_text='Distribution of Call Status', title_font=dict(size=24))

    # Display the pie chart in Streamlit
    st.plotly_chart(fig)


    # Caller Duration Analysis
    # Calculate the average, minimum, and maximum call durations
    average_duration = data['length_in_sec'].mean()
    minimum_duration = data['length_in_sec'].min()
    maximum_duration = data['length_in_sec'].max()

    st.sidebar.subheader("Call Duration Analysis")
    st.sidebar.write(f"Average call duration:", average_duration)
    st.sidebar.write(f"Minimum call duration:",minimum_duration)
    st.sidebar.write(f"Maximum call duration:",maximum_duration)
    

    # Filter the DataFrame for rows with status_name as 'Call Picked Up'
    filtered_test = data[data['status_name'] == 'Call Picked Up']

    # Calculate the average, minimum, and maximum call durations
    average_duration = filtered_test['length_in_sec'].mean()
    minimum_duration = filtered_test['length_in_sec'].min()
    maximum_duration = filtered_test['length_in_sec'].max()

    st.sidebar.subheader("Call Picked Up Analysis")
    st.sidebar.write(f"Average duration (Call Picked Up):",average_duration)
    st.sidebar.write(f"Minimum duration (Call Picked Up):", minimum_duration)
    st.sidebar.write(f"Maximum duration (Call Picked Up):", maximum_duration)

    # Caller Analysis
    # Convert 'call_date' column to datetime
    data['call_date'] = pd.to_datetime(data['call_date'])

    # Calculate the average call duration per user
    average_duration_per_user = data.groupby(['call_date', 'first_name']).agg({'length_in_sec': 'mean', 'status_name': 'first', 'phone_number': 'first', 'user': 'first'}).reset_index()

    st.title('Call Duration Analysis')

    # Add an input field for the user to enter the number of users
    num_users = st.number_input("Enter the number of users to display:", min_value=1, max_value=len(average_duration_per_user), value=10, step=1)

    # Get the top and bottom users based on the selected number
    users_highest_duration = average_duration_per_user.nlargest(num_users, 'length_in_sec').copy()
    users_lowest_duration = average_duration_per_user.nsmallest(num_users, 'length_in_sec').copy()

    # Display users with highest call durations
    st.subheader(f'Top {num_users} Users with Highest Call Durations')
    st.write(users_highest_duration[['call_date', 'user', 'phone_number', 'first_name', 'length_in_sec', 'status_name']].reset_index(drop=True))

    # Display users with lowest call durations
    st.subheader(f'Top {num_users} Users with Lowest Call Durations')
    st.write(users_lowest_duration[['call_date', 'user', 'phone_number', 'first_name', 'length_in_sec', 'status_name']].reset_index(drop=True))

    
    # Group the dataframe by phone_number, status_name, date, and time, and count the occurrences
    grouped = data.groupby(['phone_number', 'status_name', 'date', 'time']).size().reset_index(name='count')

    # Find the count of distinct phone_number occurrences
    phone_count = data.groupby('phone_number').size().reset_index(name='count')

    # Merge the original dataframe with the phone counts
    result = pd.merge(data, phone_count, on='phone_number')

    # Sort the result based on date and then phone_number
    result = result.sort_values(['date', 'phone_number'], ascending=[True, True]).reset_index(drop=True)

    # Separate 'date' and 'time' into separate columns
    result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
    result['time'] = pd.to_datetime(result['time']).dt.strftime('%H:%M:%S')

    # Streamlit app code
    st.subheader('Call Analysis Report')

    
   # Display the entire DataFrame using st.dataframe()
    columns_to_show = ['date', 'time', 'phone_number', 'user', 'first_name', 'status_name', 'list_id', 'count', 'length_in_sec']
    st.subheader("All Data")
    st.dataframe(result[columns_to_show], height=300, width=800)  # Adjust the height and width as needed

    # Allow user to select a phone number, user, and status_name after viewing the table
    phone_options = ['All'] + result['phone_number'].unique().tolist()
    selected_phone = st.selectbox("Select a phone number:", phone_options)
    status_options = ['All'] + result['status_name'].unique().tolist()
    selected_status = st.selectbox("Select a status:", status_options)

    # Allow user to select users using checkboxes
    user_options = ['All'] + result['user'].unique().tolist()
    selected_users = st.multiselect("Select users:", user_options)

    # Display details for the selected phone number, status_name(s), and users
    st.subheader(f"Selected Phone Number: {selected_phone} | Selected Status: {selected_status} | Selected Users: {', '.join(selected_users)}")

    # Filter data based on user's selections
    if selected_phone == 'All' and selected_status == 'All' and 'All' in selected_users:
        filtered_rows = result.copy()
    else:
        filtered_rows = result[
            ((result['phone_number'] == selected_phone) | (selected_phone == 'All')) &
            ((result['status_name'] == selected_status) | (selected_status == 'All')) &
            ((result['user'].isin(selected_users)) | ('All' in selected_users))
        ]

    # Reset the index of the filtered DataFrame
    filtered_rows.reset_index(drop=True, inplace=True)
    st.dataframe(filtered_rows[columns_to_show], height=300, width=800) # Adjust the height and width as needed

 
    # Convert 'time' column to datetime
    data['time'] = pd.to_datetime(data['time'])
    # Extract the hour information from the 'time' column
    data['hour'] = data['time'].dt.hour
    # Group by 'hour' and 'status_name' and calculate the count of calls
    hourly_status_count = data.groupby(['hour', 'status_name']).size().unstack().reset_index()

    # Plot the count of calls for each status at each hour using Plotly
    fig = px.bar(hourly_status_count, x='hour', y=hourly_status_count.columns[1:], 
                 title='Call Volume by Hour and Status', labels={'hour': 'Hour of the Day', 'y': 'Count of Calls'},
                 height=600)

    # Customize the layout
    fig.update_layout(legend_title_text='Status', legend=dict(x=1.02, y=1), barmode='stack', 
                      yaxis=dict(range=[0, hourly_status_count.values.max() + 10]),  # Adjust the range here
                      title=dict(text='Call Volume by Hour and Status', font=dict(size=24)))  # Adjust the title font size

    # Show the Plotly figure in Streamlit
    st.plotly_chart(fig)
    st.dataframe(hourly_status_count) 


# Footer
st.write("---")
st.write("Created By Amar Singh.")
st.write("Data Analyst")
