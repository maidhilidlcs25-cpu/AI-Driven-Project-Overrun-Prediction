import pandas as pd
import numpy as np

# Configuration
NUM_PROJECTS = 500
FILE_NAME = "powergrid_projects.csv"

# Lists for realistic data generation
project_types = ['Substation', 'Overhead Line', 'Underground Cable']
terrains = ['Plains', 'Hilly', 'Coastal', 'Forest', 'Urban']
vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D', 'Vendor E']
seasons = ['Monsoon', 'Winter', 'Summer', 'Post-Monsoon']
hindrances = ['None', 'Regulatory Delay', 'Local Protests', 'Land Acquisition', 'Extreme Weather']

# Generate data
data = {
    'project_id': [f'PG-{1001+i}' for i in range(NUM_PROJECTS)],
    'project_type': np.random.choice(project_types, NUM_PROJECTS, p=[0.3, 0.5, 0.2]),
    'terrain': np.random.choice(terrains, NUM_PROJECTS, p=[0.4, 0.2, 0.1, 0.1, 0.2]),
    'planned_cost_crores': np.random.uniform(50, 500, NUM_PROJECTS).round(2),
    'planned_timeline_months': np.random.randint(12, 48, NUM_PROJECTS),
    'material_cost_index': np.random.uniform(0.8, 1.5, NUM_PROJECTS).round(2),
    'labour_availability_percent': np.random.uniform(0.6, 1.0, NUM_PROJECTS).round(2),
    'vendor': np.random.choice(vendors, NUM_PROJECTS),
    'season_of_start': np.random.choice(seasons, NUM_PROJECTS),
    'primary_hindrance': np.random.choice(hindrances, NUM_PROJECTS, p=[0.5, 0.15, 0.1, 0.15, 0.1]),
    
    # Target Variables (what we want to predict)
    'actual_cost_crores': [0] * NUM_PROJECTS,
    'actual_timeline_months': [0] * NUM_PROJECTS,
    'cost_overrun_percent': [0.0] * NUM_PROJECTS,
    'timeline_overrun_percent': [0.0] * NUM_PROJECTS,
}

df = pd.DataFrame(data)

# Simulate realistic overruns based on factors
# Higher costs/delays for Hilly terrain, Monsoon, certain vendors, etc.
cost_overrun = df['planned_cost_crores'] * (1 + (
    (df['terrain'] == 'Hilly').astype(int) * np.random.uniform(0.1, 0.3) +
    (df['terrain'] == 'Forest').astype(int) * np.random.uniform(0.05, 0.15) +
    (df['vendor'] == 'Vendor C').astype(int) * np.random.uniform(0.05, 0.1) +
    (df['primary_hindrance'] == 'Regulatory Delay').astype(int) * np.random.uniform(0.1, 0.25) +
    (df['material_cost_index'] - 1) * 0.5 +
    (1 - df['labour_availability_percent']) * 0.3
)).round(2)

timeline_overrun = df['planned_timeline_months'] * (1 + (
    (df['terrain'] == 'Hilly').astype(int) * np.random.uniform(0.2, 0.5) +
    (df['season_of_start'] == 'Monsoon').astype(int) * np.random.uniform(0.1, 0.3) +
    (df['vendor'] == 'Vendor C').astype(int) * np.random.uniform(0.1, 0.2) +
    (df['primary_hindrance'] != 'None').astype(int) * np.random.uniform(0.15, 0.4) +
    (1 - df['labour_availability_percent']) * 0.4
)).round(0)

df['actual_cost_crores'] = cost_overrun.round(2)
df['actual_timeline_months'] = timeline_overrun.astype(int)
df['cost_overrun_percent'] = ((df['actual_cost_crores'] / df['planned_cost_crores'] - 1) * 100).round(2)
df['timeline_overrun_percent'] = ((df['actual_timeline_months'] / df['planned_timeline_months'] - 1) * 100).round(2)

# Save to CSV
df.to_csv(FILE_NAME, index=False)

print(f"Successfully generated synthetic data at '{FILE_NAME}'")
print(df.head())
