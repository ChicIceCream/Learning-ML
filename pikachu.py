import matplotlib.pyplot as plt
import numpy as np

# Define Pikachu body
def draw_pikachu(ax):
    # Body
    body_x = np.linspace(0, 2*np.pi, 100)
    body_y = np.sin(body_x)
    ax.plot(0.5 + 0.2*body_x/np.pi, 0.5 + 0.2*body_y, color='yellow', linewidth=5)

    # Ears
    ear1_x = np.linspace(0, np.pi, 100)
    ear1_y = np.sin(ear1_x)
    ax.plot(0.4 + 0.05*ear1_x/np.pi, 0.8 + 0.05*ear1_y, color='yellow', linewidth=5)
    ear2_x = np.linspace(np.pi, 2*np.pi, 100)
    ear2_y = np.sin(ear2_x)
    ax.plot(0.6 + 0.05*(ear2_x-np.pi)/np.pi, 0.8 + 0.05*ear2_y, color='yellow', linewidth=5)

    # Eyes
    eye1 = plt.Circle((0.45, 0.6), 0.02, color='black')
    eye2 = plt.Circle((0.55, 0.6), 0.02, color='black')
    ax.add_artist(eye1)
    ax.add_artist(eye2)

    # Cheeks
    cheek1 = plt.Circle((0.35, 0.5), 0.05, color='red', alpha=0.5)
    cheek2 = plt.Circle((0.65, 0.5), 0.05, color='red', alpha=0.5)
    ax.add_artist(cheek1)
    ax.add_artist(cheek2)

    # Mouth
    mouth_x = np.linspace(0.4, 0.6, 100)
    mouth_y = -0.1 * (mouth_x - 0.5)**2 + 0.55
    ax.plot(mouth_x, mouth_y, color='black', linewidth=3)

    # Adding years
    ax.text(0.5, 0.2, "5 years old", ha='center', fontsize=12, fontweight='bold')

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Draw Pikachu
draw_pikachu(ax)

# Background
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')

# Show plot
plt.show()
