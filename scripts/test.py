import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example DataFrame
data = {
    'feature1': ['A', 'B', 'A', 'B', 'A', 'B'],
    'feature2': ['X', 'X', 'Y', 'Y', 'X', 'X'],
    'feature3': ['P', 'P', 'Q', 'Q', 'Q', 'Q'],
    'target': [10, 20, 30, 40, 50, 60]
}

df = pd.DataFrame(data)

# Create pivot table
pivot_table = df.pivot_table(index='feature1', columns=['feature2', 'feature3'], values='target', aggfunc='mean')

# Print the pivot table
print(pivot_table)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar=True)
plt.title('Heatmap of Average Target Values')
plt.show()
