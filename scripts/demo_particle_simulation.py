import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

# Constants
NUM_PARTICLES = 50
TIMESTEPS = 500
DT = 0.01  # Time step
SPACE_SIZE = 10.0  # Size of the simulation box

# Initialize particle positions and velocities
positions = np.random.uniform(-SPACE_SIZE, SPACE_SIZE, (NUM_PARTICLES, 2))
old_positions = positions - np.random.uniform(
    -0.1, 0.1, (NUM_PARTICLES, 2)
)  # Small random velocities


# Forces between particles (attractive or repulsive)
def compute_forces(positions):
    forces = np.zeros_like(positions)
    for i in range(NUM_PARTICLES):
        for j in range(i + 1, NUM_PARTICLES):
            # Compute distance vector and squared distance
            r_vec = positions[j] - positions[i]
            r2 = np.dot(r_vec, r_vec)

            # Avoid division by zero
            if r2 == 0:
                continue

            # Force magnitude (change sign for attractive/repulsive behavior)
            force_magnitude = 1 / r2 if r2 > 1e-4 else 0

            # Repulsive force
            force_direction = r_vec / np.sqrt(r2)
            force = force_magnitude * force_direction

            # Apply force symmetrically
            forces[i] += force
            forces[j] -= force
    return forces


# Verlet integration
trajectories = []
for t in range(TIMESTEPS):
    forces = compute_forces(positions)

    # Verlet position update
    new_positions = 2 * positions - old_positions + forces * DT**2

    # Save trajectory for animation
    trajectories.append(new_positions.copy())

    # Update positions
    old_positions = positions
    positions = new_positions

# Convert trajectories to array for easier access
trajectories = np.array(trajectories)

# Animation
fig, ax = plt.subplots()
(points,) = ax.plot([], [], "bo", markersize=4)

ax.set_xlim(-SPACE_SIZE, SPACE_SIZE)
ax.set_ylim(-SPACE_SIZE, SPACE_SIZE)
ax.set_aspect("equal")
ax.set_title("Particle Simulation")


# Initialize the plot
def init():
    points.set_data([], [])
    return (points,)


# Update the animation for each frame
def update(frame):
    points.set_data(trajectories[frame, :, 0], trajectories[frame, :, 1])
    return (points,)


# Animate
anim = FuncAnimation(fig, update, frames=TIMESTEPS, init_func=init, blit=True)

# Save the animation
writer = FFMpegWriter(fps=30, metadata=dict(artist="Me"), bitrate=1800)
anim.save("particle_simulation.mp4", writer=writer)

plt.show()
