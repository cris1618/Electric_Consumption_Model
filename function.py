import numpy as np
import matplotlib.pyplot as plt

# 1. Create an array of x-values from 0 to pi/2
x = np.linspace(0, 2*np.pi, 400)

# 2. Define the function: sin(x) - cos(2x)
y1 = np.cos(x)
y2 = x + 2*np.sin(x)**4

# 3. Identify the zero crossing near x = pi/6 for reference
x_cross = np.pi/6  # We’ve already found sin(x) = 1/2 is the point of sign change
y_cross = 0        # zero at that point

# 4. Create the figure and axis
fig, ax = plt.subplots(figsize=(7, 4))
plt.axhline(0, color='gray', linewidth=1)  # horizontal axis

# 5. Plot the function
ax.plot(x, y1, color='blue')
ax.plot(x, y2, color="red")

# 7. Format the axes and add labels
plt.xlim(0, 2*np.pi)
#plt.ylim(-10, 10)
plt.grid(True, alpha=0.3)

# 8. Show the plot
plt.show()
