import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files without headers
force_control_data = pd.read_csv('force_control_1.csv', header=None)
impedance_control_data = pd.read_csv('impedance_control_1.csv', header=None)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(force_control_data[0], force_control_data[1], label='Force Control', color='blue')
plt.plot(impedance_control_data[0], impedance_control_data[1], label='Impedance Control', color='red')

# Customize the plot
plt.ylim(10,20)
plt.xlim(0,0.4)
plt.title('Force Control vs Impedance Control (Static Board)')
plt.xlabel('Time')
plt.ylabel('Force')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
