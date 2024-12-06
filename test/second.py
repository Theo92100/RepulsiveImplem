import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Create a grid for the plane
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)  # Plane at z = 0

# Define collision region (a circle centered at (0,0) with radius 1.5)
collision_center = (0, 0)
collision_radius = 1.5
collision_mask = (X - collision_center[0])**2 + (Y - collision_center[1])**2 <= collision_radius**2

# First Manifold: Plane with highlighted collision region
# Create a custom colormap that uses a warm color for the collision region
from matplotlib.colors import ListedColormap

# Define colors: blue for normal region, red for collision region
colors = np.array([
    [0.2, 0.4, 0.6, 1],  # Blue color for the plane
    [1, 0.5, 0, 1]       # Warm color (orange) for collision region
])

# Create a colormap
cmap = ListedColormap(colors)

# Create an array to map colors
color_map_array = np.zeros_like(Z)
color_map_array[collision_mask] = 1  # Set collision region to 1 (warm color)

# Plot the first manifold
fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# Plot the surface with the custom colormap
surf1 = ax1.plot_surface(X, Y, Z, facecolors=cmap(color_map_array), edgecolor='none')

# Remove axes
ax1.set_axis_off()

# Set view angle for better visualization
ax1.view_init(elev=90, azim=-90)  # Top-down view

plt.tight_layout()
plt.show()

# Second Manifold: Plane with a peak at the collision region
# Define a peak function (Gaussian)
def peak_function(X, Y, center, radius, height):
    return height * np.exp(-(((X - center[0])**2 + (Y - center[1])**2) / (2 * (radius/3)**2)))

Z_peak = Z + peak_function(X, Y, collision_center, collision_radius, height=2)

# Plot the second manifold
fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# Plot the surface
surf2 = ax2.plot_surface(X, Y, Z_peak, cmap='viridis', edgecolor='none')
fig2.text(0.5, 0.95, r'Manifold $M^\phi$ with Infinite Tall Mountains', ha='center', va='top', fontsize=14)

# Remove axes
ax2.set_axis_off()

# Set view angle for better visualization
ax2.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.show()
