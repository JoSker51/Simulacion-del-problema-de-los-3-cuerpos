import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

G = 9.8
N = 4  #este es el numero de cuerpos
delta_t = 0.001
steps = 50000

# masas aleatorias
masses = np.random.uniform(5, 30, N)

# posiciones y velocidades iniciales aleatorias
positions = np.random.uniform(-20, 20, (N, 3))
velocities = np.random.uniform(-2, 2, (N, 3))

# almacenar trayectorias
traj = np.zeros((steps, N, 3))
traj[0] = positions


def compute_accelerations(pos, masses):
    acc = np.zeros_like(pos)
    for i in range(N):
        for j in range(N):
            if i != j:
                r = pos[i] - pos[j]
                dist = np.linalg.norm(r) + 1e-5
                acc[i] += -G * masses[j] * r / dist**3
    return acc

#agarrar entropia
def compute_entropy(pos, bins=10):
    hist, _ = np.histogramdd(pos, bins=bins)
    p = hist.flatten()
    p = p[p > 0]
    p = p / np.sum(p)
    return -np.sum(p * np.log(p))

entropy_values = []


for t in range(steps - 1):
    acc = compute_accelerations(positions, masses)
    velocities += acc * delta_t
    positions += velocities * delta_t
    
    traj[t + 1] = positions
    entropy_values.append(compute_entropy(positions))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

for i in range(N):
    ax.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], lw=0.5)

ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
plt.show()

#graficar
plt.figure()
plt.plot(entropy_values)
plt.title("Entropía del sistema")
plt.xlabel("Tiempo")
plt.ylabel("S")
plt.show()
