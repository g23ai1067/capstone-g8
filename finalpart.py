import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
working_dataset = r'C:\Users\bruha\OneDrive\Desktop\Capstone_Project\HotelBookings Dataset.csv'
df = pd.read_csv(working_dataset)

# View first and last 5 rows
print(df.head())
print(df.tail())

# Check rows and columns count
print("Shape of dataset:", df.shape)

# Check dataset information
print(df.info())

# Check duplicate values
duplicate_values = df.duplicated().sum()
print("Number of duplicate rows:", duplicate_values)

# Visualize duplicate values
plt.figure(figsize=(10, 6))
sns.countplot(x=df.duplicated())
plt.title('Visualization of Duplicate Values')
plt.xlabel('Duplicate Status')
plt.ylabel('Count')
plt.show()

# Remove duplicate rows
df = df.drop_duplicates()

# Check missing values
missing_values = df.isnull().sum().sort_values(ascending=False)
print("Missing values:\n", missing_values.head())

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title('Visualization of Missing Values')
plt.show()

# Handle missing values
df['company'].fillna(0, inplace=True)
df['agent'].fillna(0, inplace=True)
df['children'].fillna(0, inplace=True)
df['country'].fillna('Others', inplace=True)

# Drop rows where no guests are listed
df = df[(df['adults'] + df['children'] + df['babies']) > 0]

# Add additional columns
df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['total_people'] = df['adults'] + df['children'] + df['babies']

# Example: Visualizing preferred hotel type
plt.figure(figsize=(10, 6))
sns.countplot(x='hotel', data=df)
plt.title('Preferred Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Count')
plt.show()

# Example: Average ADR for each hotel type
plt.figure(figsize=(10, 6))
sns.barplot(x='hotel', y='adr', data=df)
plt.title('Average ADR for Each Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('ADR')
plt.show()

# Hotel with maximum bookings
max_bookings_hotel = df['hotel'].value_counts().idxmax()
max_bookings_count = df['hotel'].value_counts().max()

print(f"The hotel with maximum bookings is {max_bookings_hotel} with {max_bookings_count} bookings.")

# Agent with maximum bookings
max_bookings_agent = df['agent'].value_counts().idxmax()
max_bookings_agent_count = df['agent'].value_counts().max()

print(f"The agent with maximum bookings is {max_bookings_agent} with {max_bookings_agent_count} bookings.")

# Percentage of repeated guests
total_guests = df.shape[0]
repeated_guests = df[df['is_repeated_guest'] == 1].shape[0]
percentage_repeated_guests = (repeated_guests / total_guests) * 100

print(f"The percentage of repeated guests is {percentage_repeated_guests:.2f}%.")


# Example: Visualizing preferred room type
plt.figure(figsize=(10, 6))
sns.countplot(x='reserved_room_type', data=df)
plt.title('Preferred Room Type')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.show()

# Example: Visualizing preferred meal choice
plt.figure(figsize=(10, 6))
sns.countplot(x='meal', data=df)
plt.title('Preferred Meal Choice')
plt.xlabel('Meal Type')
plt.ylabel('Count')
plt.show()



# Example: Visualizing bookings by month
plt.figure(figsize=(12, 6))
sns.countplot(x='arrival_date_month', data=df, palette='viridis')
plt.title('Bookings by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Find month with the most bookings
most_bookings_month = df['arrival_date_month'].value_counts().idxmax()
most_bookings_month_count = df['arrival_date_month'].value_counts().max()

print(f"The month with the most bookings is '{most_bookings_month}' with {most_bookings_month_count} bookings.")


# Visualize distribution of bookings by distribution channel
plt.figure(figsize=(10, 6))
sns.countplot(x='distribution_channel', data=df, palette='viridis')
plt.title('Bookings by Distribution Channel')
plt.xlabel('Distribution Channel')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate percentage of bookings for each distribution channel
channel_counts = df['distribution_channel'].value_counts(normalize=True) * 100

print("Percentage of bookings by distribution channel:")
print(channel_counts)


# Find year with highest bookings
df['arrival_date'] = pd.to_datetime(df['arrival_date'])
df['arrival_year'] = df['arrival_date'].dt.year

plt.figure(figsize=(10, 6))
sns.countplot(x='arrival_year', data=df, palette='Set2')
plt.title('Bookings per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

year_with_most_bookings = df['arrival_year'].value_counts().idxmax()
print(f"Year with the most bookings: {year_with_most_bookings}")

# Find hotel with most wait time
hotel_wait_time = df.groupby('hotel')['days_in_waiting_list'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=hotel_wait_time.index, y=hotel_wait_time.values, palette='Set1')
plt.title('Average Wait Time by Hotel')
plt.xlabel('Hotel')
plt.ylabel('Average Wait Time (days)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

hotel_with_most_wait_time = hotel_wait_time.idxmax()
avg_wait_time = hotel_wait_time.max()
print(f"Hotel with the most wait time: {hotel_with_most_wait_time}")
print(f"Average wait time: {avg_wait_time:.2f} days")

# Calculate average ADR by distribution channel
plt.figure(figsize=(10, 6))
sns.barplot(x='distribution_channel', y='adr', data=df, estimator=np.mean, ci=None, palette='Blues_d')
plt.title('Average Daily Rate (ADR) by Distribution Channel')
plt.xlabel('Distribution Channel')
plt.ylabel('Average Daily Rate (ADR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df_not_canceled = df[df['is_canceled'] == 0]

# Visualize relationship between repeated guests and previous bookings not canceled
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_repeated_guest', y='previous_bookings_not_canceled', data=df_not_canceled)
plt.title('Relationship between Repeated Guests and Previous Bookings Not Canceled')
plt.xlabel('Repeated Guest')
plt.ylabel('Previous Bookings Not Canceled')
plt.xticks([0, 1], ['Not Repeated', 'Repeated'])
plt.tight_layout()
plt.show()

# Statistical summary
summary = df_not_canceled.groupby('is_repeated_guest')['previous_bookings_not_canceled'].describe()
print("Statistical summary of previous bookings not canceled:")
print(summary)

# Create correlation matrix for numeric columns
numeric_cols = df_not_canceled.select_dtypes(include=['number']).columns
corr_matrix = df_not_canceled[numeric_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Features in Hotel Bookings Dataset')
plt.show()