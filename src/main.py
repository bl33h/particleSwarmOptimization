import numpy as np
import matplotlib.pyplot as plt

# objective function definition
def function(position):
    x, y = position
    return (x - 3)**2 - (y - 2)**2

# function to create contour plot
def plotContour(ax):
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = function((X, Y))
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("PSO Optimization")

# initialize parameters
particlesNumber = 200  # number of particles
iterationsQ = 100  # number of iterations
w = 0.5  # inertia weight
c1 = 2.0  # personal acceleration coefficient
c2 = 2.0  # global acceleration coefficient
bounds = [(-10, 10), (-10, 10)]  # search space bounds for (x, y)

# initialize particle positions and velocities
particles = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (particlesNumber, 2))
velocities = np.random.uniform(-1, 1, (particlesNumber, 2))

# initialize personal and global bests
personalBestPositions = np.copy(particles)
personalBestScores = np.apply_along_axis(function, 1, particles)
actualBestPosition = personalBestPositions[np.argmin(personalBestScores)]
actualBestScore = np.min(personalBestScores)

# initialize plot
plt.ion()
fig, ax = plt.subplots()
plotContour(ax)
sc = ax.scatter(particles[:, 0], particles[:, 1], color='white', edgecolor='black')
globalBestScatter = ax.scatter(actualBestPosition[0], actualBestPosition[1], color='red')
plt.title("PSO Optimization")

def updatePlot(particles, actualBestPosition):
    sc.set_offsets(particles)
    globalBestScatter.set_offsets(actualBestPosition)
    plt.draw()
    plt.pause(0.1)

# PSO algorithm main loop
for iteration in range(iterationsQ):
    for i in range(particlesNumber):
        # update velocity
        r1, r2 = np.random.rand(2)
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (personalBestPositions[i] - particles[i]) +
                         c2 * r2 * (actualBestPosition - particles[i]))
        
        # update position
        particles[i] += velocities[i]
        
        # constrain particles within bounds
        particles[i] = np.clip(particles[i], [b[0] for b in bounds], [b[1] for b in bounds])
        
        # evaluate new position
        score = function(particles[i])
        
        # update personal best
        if score < personalBestScores[i]:
            personalBestPositions[i] = particles[i]
            personalBestScores[i] = score
    
    # update global best
    currentBestScore = np.min(personalBestScores)
    if currentBestScore < actualBestScore:
        actualBestPosition = personalBestPositions[np.argmin(personalBestScores)]
        actualBestScore = currentBestScore
    
    # update plot
    updatePlot(particles, actualBestPosition)
    
    print(f"Iteration {iteration + 1}/{iterationsQ}, Global Best Score: {actualBestScore}")

print("Global Best Position:", actualBestPosition)
print("Global Best Score:", actualBestScore)
plt.ioff()
plt.show()
