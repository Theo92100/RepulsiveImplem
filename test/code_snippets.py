import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Create a grid for the plane
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)  # Plane at z = 0

# Define collision region (a circle centered at (0,0) with radius 1.5)
collision_center = (0, 0)
collision_radius = 1.5
collision_mask = (X - collision_center[0])**2 + (Y - collision_center[1])**2 <= collision_radius**2

### First Manifold: Plane with highlighted collision region (M) ###

# Define colors: blue for normal region, orange for collision region
colors = np.array([
    [0.2, 0.4, 0.6, 1],  # Blue color for the plane (normal region)
    [1, 0, 0, 1]       # Orange color for the collision region
])

# Create a custom colormap
cmap = ListedColormap(colors)

# Create an array to map colors
color_map_array = np.zeros_like(Z)
color_map_array[collision_mask] = 1  # Set collision region to 1 (orange)

# Plot the first manifold
fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# Plot the surface with the custom colormap
surf1 = ax1.plot_surface(X, Y, Z, facecolors=cmap(color_map_array), edgecolor='none')

# Remove axes
ax1.set_axis_off()

# Set view angle for better visualization
ax1.view_init(elev=30, azim=-60)  # Top-down view

# Add legend
legend_elements = [Patch(facecolor=colors[0], edgecolor='none', label='M (Manifold)'),
                   Patch(facecolor=colors[1], edgecolor='none', label='Collision Region')]
fig1.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

# Add title
fig1.text(0.5, 0.95, 'Manifold M with Collision Region', ha='center', va='top', fontsize=14)

plt.tight_layout()
plt.show()

### Second Manifold: Plane with "infinite tall mountains" at the collision region (M^φ) ###

# Set a very high value for the collision region to simulate infinite peaks
Z_inf = np.copy(Z)
Z_inf[collision_mask] = 10  # Set the collision region to a high Z value

# Create a color map
# Reuse the colors array
cmap_inf = ListedColormap(colors)

# Create an array to map colors
color_map_array_inf = np.zeros_like(Z_inf)
color_map_array_inf[collision_mask] = 1  # Set infinite tall mountains to 1 (orange)

# Plot the second manifold
fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# Plot the surface with the custom colormap
surf2 = ax2.plot_surface(X, Y, Z_inf, facecolors=cmap_inf(color_map_array_inf), edgecolor='none')

# Remove axes
ax2.set_axis_off()

# Set view angle for better visualization
ax2.view_init(elev=30, azim=-60)

# Adjust Z-axis limits to show the peak clearly
ax2.set_zlim(0, 10)

# Add legend
legend_elements_inf = [Patch(facecolor=colors[0], edgecolor='none', label=r'$M^\phi$ (Graph Manifold)'),
                       Patch(facecolor=colors[1], edgecolor='none', label='Infinite Tall Mountains')]
fig2.legend(handles=legend_elements_inf, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

# Add title
fig2.text(0.5, 0.95, r'Manifold $M^\phi$ with Infinite Tall Mountains', ha='center', va='top', fontsize=14)

plt.tight_layout()
plt.show()
