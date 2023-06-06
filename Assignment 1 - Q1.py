import numpy as np
import matplotlib.pyplot as plt

# Custom function to calculate chi-squared PDF
def chi_squared_pdf(x, k):
    return (1 / (2 ** (k / 2) * np.math.gamma(k / 2))) * (x ** (k / 2 - 1)) * np.exp(-x / 2)

# Set up parameters
n = 10**4
colors = ['r', 'g', 'b', 'c', 'y']
kk = [1, 2, 3, 10, 100]

# Plotting settings
plt.xlabel('Data points')
plt.ylabel('Probability Density')
plt.title('Custom Plot')
plt.xlim(0, 150)
plt.ylim(0, 0.3)

# Generate x values
x = np.linspace(0, 150, n)

# Plot each distribution
for k_value, color in zip(kk, colors):
    y = chi_squared_pdf(x, k_value)
    plt.plot(x, y, color, label='k=mu={:.0f} , sd={:.2f}'.format(k_value, 2 * k_value))

plt.legend()
plt.show()
