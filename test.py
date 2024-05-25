import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Constants
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
current = 75  # Current in the solenoid (A)
L = 0.5  # Length of the solenoid in meters
tolerance = 0.05  # 5% tolerance

class Mag_Field_equations:
    def __init__(self, L, I, mu_0):
        self.L = L 
        self.I = I
        self.mew_0 = mu_0
    
    def _get_B_center(self, turns):
        n = turns / self.L  # We calculate the coil density (number of turns / length of solenoid)
        B_center = self.mew_0 * n * self.I
        return B_center
    
    def _get_mag_field_z(self, z, turns):
        n = turns / L  # We calculate the coil density (number of turns / length of solenoid)
        return self.mew_0 * n * self.I * (1 - z**2 / self.L**2)
    
    def _get_avg_mag_field(self, turns):
        z_pts = np.linspace(-self.L/2, self.L/2, 1000)
        B_pts = [B_z(z, turns) for z in z_pts]
        return np.mean(B_pts)


# Function to calculate the magnetic field at the center of a solenoid using Biot-Savart law
def B_center(N):
    n = N / L  # Coil density
    B_c = mu_0 * n * current
    return B_c

# Function to calculate the magnetic field at a point z inside the solenoid using Biot-Savart law
def B_z(z, N):
    n = N / L  # Coil density
    B = mu_0 * n * current * (1 - z**2 / L**2)
    return B

# Function to calculate the average magnetic field in the interior region of the solenoid
def B_avg(N):
    z_points = np.linspace(-L/2, L/2, 1000)
    B_points = [B_z(z, N) for z in z_points]
    return np.mean(B_points)

# Function to check if the magnetic field is within tolerance
def is_within_tolerance(N):
    B_c = B_center(N)
    B_avg_val = B_avg(N)
    return abs(B_avg_val - B_c) / B_c <= tolerance

# Golden section search for a single variable
def golden_section_search(f, a, b, tol=1e-5):
    gr = (np.sqrt(5) + 1) / 2  # Golden ratio
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2

# Objective function for the number of turns optimization
def objective_N(N):
    B_avg_val = B_avg(N)
    B_c = B_center(N)
    return abs(B_avg_val - B_c) / B_c

# Function to dynamically determine the bounds for the golden section search
def find_initial_bounds():
    N = 1
    while not is_within_tolerance(N):
        N *= 2
        if N > 1e6:  # Preventing too large numbers
            break
    lower_bound = N // 2 if N < 1e6 else N // 4
    upper_bound = N if N < 1e6 else N // 2
    return lower_bound, upper_bound

# Find the initial bounds
a_N, b_N = find_initial_bounds()

# Optimize for the minimum number of turns
optimal_N = golden_section_search(objective_N, a_N, b_N)

# Print the optimal number of coils
print(f"The optimal number of coils (N) for a uniform magnetic field within 5% tolerance is: {optimal_N}")

# Calculate the magnetic fields for plotting
N_values = np.linspace(a_N, b_N, 500)
B_avg_values = [B_avg(N) for N in N_values]
B_center_values = [B_center(N) for N in N_values]

# Calculate the 5% tolerance bands
B_center_optimal = B_center(optimal_N)
tolerance_band_upper = B_center_optimal * (1 + tolerance)
tolerance_band_lower = B_center_optimal * (1 - tolerance)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(N_values, B_avg_values, label='Average Magnetic Field in Solenoid')
plt.plot(N_values, B_center_values, label='Magnetic Field at Center')
plt.axhline(y=tolerance_band_upper, color='r', linestyle='--', label='5% Tolerance Band')
plt.axhline(y=tolerance_band_lower, color='r', linestyle='--')
plt.axvline(x=optimal_N, color='g', linestyle='--', label='Optimal Number of Turns')
plt.xlabel('Number of Turns (N)')
plt.ylabel('Magnetic Field (T)')
plt.title('Magnetic Field in Solenoid vs. Number of Turns')
plt.legend()
plt.grid(True)
plt.show()
