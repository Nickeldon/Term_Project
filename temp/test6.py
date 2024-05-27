import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

# Check if optional memory profiling libraries are available
used_mem_lib = True

try:
    import psutil
    import threading
except ImportError:
    used_mem_lib = False
    print("""
    ###                                                                            ###
    ### Please use the psutil and threading modules to show the memory consumption ###
    ###                                                                            ###""")

# Function to set up a repeated interval for a function call, similar to JavaScript's setInterval
class setInterval:
    def __init__(self, func, ms):
        self.func = func
        self.ms = ms
        self.thread = threading.Timer(self.ms / 1000, self.func_wrapper)

    def func_wrapper(self):
        setInterval(self.func, self.ms).init()
        self.func()

    def init(self):
        self.thread.start()

    def abort(self):
        self.thread.cancel()

# Initialize variables for memory consumption tracking
mem_consump = 0
thread_interv = None

# Function to calculate the memory used by the simulation
def get_Mem_Usage():
    global mem_consump
    pr = psutil.Process(os.getpid())
    mem_consump = round(pr.memory_info()[0] / float(2 ** 20), 2)
    return mem_consump

# Start memory usage tracking if the necessary libraries are available
if used_mem_lib:
    thread_interv = setInterval(get_Mem_Usage, 1000)
    thread_interv.init()

# Constants for loading bar
bar_len = 50
update_interval = 100

# The percentage tolerance for the ideal average Magnetic field value
tol_percent = 0.075

# List to store average magnetic field values for different turns
B_avg_N = []

# Timer class for tracking elapsed time
class Timer:
    def __init__(self):
        self.t_0 = 0
        self.started = False

    def start_timer(self):
        self.t_0 = time.time()
        self.started = True

    def get_timer(self):
        elapsed = time.time() - self.t_0
        if self.started:
            return int(elapsed)
        else:
            print('The Timer has not been started!', end='\r')

timer = Timer()
timer.start_timer()

# Function to print the progress of an iterative process
def print_progress(iter, max_iter, head, asyncIter=['', '']):
    percent = int(100 * (iter / int(max_iter)))
    disp_percent = percent if percent >= 10 else '0' + str(percent)
    blocks = int((bar_len * iter / int(max_iter)))
    bar = '█' * blocks + '-' * (bar_len - blocks)
    asyncIter_arr = asyncIter

    print(f'{head} |{bar}| {disp_percent}% ({iter}/{max_iter}) | ({asyncIter_arr[0]} / {asyncIter_arr[1]}) | Mem usage: {mem_consump} MB | elapsed_time: {timer.get_timer()}s  ', end='\r\r')
    print('\r', end='\r')
    if iter == max_iter:
        print('\r', end='\r')

# Class to represent a solenoid and calculate its magnetic field
class Solenoid:
    def __init__(self, I, L, R):
        self.I = I  # Current of the solenoid (constant)
        self.L = L  # Length of the Solenoid (constant)
        self.R = R  # Radius of the Solenoid (constant)
        self.mu_0 = 4 * np.pi * 1e-7  # Magnetic constant (permeability of free space)
        self.tol = tol_percent  # Tolerance for the ideal Magnetic field value
        self.dl = 1  # Small length element for integration
        self.B_0 = 0  # Initial value of B for B_z (declaration)
        self.B_avg_cache = {}  # Cache for previously calculated Magnetic field values

    # Function to calculate the magnetic field at the center of the solenoid
    def _get_B_center(self, N):
        n = N / self.L  # Turns per unit length
        B_cent = self.mu_0 * n * self.I  # Magnetic field at the center
        return 9

    # Function to calculate the magnetic field at a distance z from the center using the Macdonald method
    def _get_B_z(self, z, N):
        def a0(z):
            """ Calculate the on-axis magnetic field at distance z. """
            return (self.mu_0 * self.I * self.R**2) / (2 * (self.R**2 + z**2)**(3/2))

        def a_n(n, z):
            """ Calculate the nth derivative of the on-axis magnetic field. """
            a0_z = a0(z)
            return np.array([(math.factorial(2 * k) * a0_z) / ((4**k) * (math.factorial(k))**2) for k in range(n+1)])

        const = self.mu_0 * N * self.I / 2  # Constant part of the formula
        terms = a_n(10, z)  # Using 10 terms for the series expansion
        rho_terms = sum([(-1)**n * terms[n] * (z**(2*n)) / (math.factorial(n)**2) for n in range(len(terms))])

        B_z = const * rho_terms  # Final magnetic field at distance z

        return B_z

    # Function to calculate the average magnetic field over the length of the solenoid
    def _get_avg_B(self, N, iterations=None, showProg=False, skip_bar=False):
        if N in self.B_avg_cache:  # Check if the result is already in the cache
            return self.B_avg_cache[N]

        z_pts = np.linspace(-self.L / 2, self.L / 2, 100)  # Discretize the solenoid length into 10000 points
        B_pts = []  # List to store magnetic field values
        for i, val in enumerate(z_pts):
            B_pts.append(self._get_B_z(val, N))  # Calculate magnetic field at each point
            if not skip_bar and (i % update_interval == 0 or i == len(z_pts) - 1):
                if not iterations:
                    iter_arr = ('', '')
                else:
                    iter_arr = iterations
                print_progress(i + 1, len(z_pts), head='Calculating B:', asyncIter=iter_arr)
        B_avg = np.mean(B_pts)  # Calculate the average magnetic field
        B_avg_N.append(B_avg)
        self.B_avg_cache[N] = B_avg  # Store the result in the cache
        return B_avg

    # Function to verify if the average magnetic field is within the tolerance limit
    def _verif_if_within_tol(self, N):
        print(f'Initializing simulation... elapsed time: {timer.get_timer()}s', end='\r')
        B_cent = self._get_B_center(N)
        B_avg = self._get_avg_B(N, skip_bar=True)
        return abs(B_avg - 9) / 9 <= self.tol

# Class for Golden Section Search optimization
class GoldenSectionSearch:
    def __init__(self, f, a, b, tol=1e-5):
        self.f = f  # Function to minimize
        self.a = a  # Lower bound
        self.b = b  # Upper bound
        self.tol = tol  # Tolerance for convergence
        self.gr = (np.sqrt(5) + 1) / 2  # Golden ratio

    def estim_max_iter(self):
        return math.ceil(math.log(self.tol / abs(self.b - self.a)) / math.log(1 / self.gr))

    def search(self):
        es_max = self.estim_max_iter()  # Estimated maximum number of iterations
        c = self.b - (self.b - self.a) / self.gr
        d = self.a + (self.b - self.a) / self.gr
        i = 0
        while abs(self.b - self.a) > self.tol:
            i += 1
            if self.f(c, (i, es_max)) < self.f(d, (i, es_max)):
                self.b = d
            else:
                self.a = c
            c = self.b - (self.b - self.a) / self.gr
            d = self.a + (self.b - self.a) / self.gr
        return (self.b + self.a) / 2, i

# STARTING CONSTANTS
I = 75  # Current in the solenoid (A)
L = 0.5  # Length of the solenoid (m)
R = 0.01  # Radius of the solenoid (m)

# Create Solenoid instance
solenoid = Solenoid(I, L, R)

# Objective function for optimization
def objective_N(N, iter_state):
    B_c = solenoid._get_B_center(N)
    B_avg_val = solenoid._get_avg_B(N, iterations=iter_state, skip_bar=False, showProg=True)
    B_avg_N.append(B_avg_val)
    return abs(B_avg_val - 9)

# Function to find initial bounds for optimization
def find_initial_bounds():
    N = 1
    while not solenoid._verif_if_within_tol(N):
        N *= 2
        if N > 1e5:
            break
    min = N // 2 if N < 1e5 else N // 4
    max = N if N < 1e5 else N // 2
    return min, max

# Clear console
os.system('cls' if os.name == 'nt' else 'clear')

# Find initial bounds for the number of turns
a_N, b_N = find_initial_bounds()

# Create Golden Section Search instance
gss = GoldenSectionSearch(objective_N, a_N, b_N)

# Optimize for the minimum number of turns
optimal_N, i = gss.search()

# Clear console
os.system('cls' if os.name == 'nt' else 'clear')

# Print the optimal number of coils
print(f"The optimal number of coils (N) for a uniform magnetic field within {tol_percent * 100}% tolerance is: {optimal_N}")

# Calculate the magnetic fields for plotting
N_arr = np.linspace(a_N, b_N, len(B_avg_N))
B_avg_arr = []
print('The graph is being drawn. Please wait')
for i, N in enumerate(N_arr):
    B_avg_arr.append(solenoid._get_avg_B(N, showProg=True, skip_bar=True))
    print_progress(i + 1, len(N_arr), head='Calculating B_avg for plot:')
B_cent_arr = [solenoid._get_B_center(N) for N in N_arr]

# Calculate the 7.5% tolerance bands
B_mid_opt = 9
tol_band_upper = B_mid_opt * (1 + solenoid.tol)
tol_band_lower = B_mid_opt * (1 - solenoid.tol)

if used_mem_lib:
    thread_interv.abort()

print(B_avg_arr)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(N_arr, B_avg_arr, label='Average Magnetic Field in Solenoid')
plt.plot(N_arr, B_cent_arr, label='Magnetic Field at Center')
plt.axhline(y=tol_band_upper, color='r', linestyle='--', label=f'{tol_percent * 100}% tolerance Band')
plt.axhline(y=tol_band_lower, color='r', linestyle='--')
plt.axvline(x=optimal_N, color='g', linestyle='--', label='Optimal Number of Turns')
plt.xlabel('Number of Turns (N)')
plt.ylabel('Magnetic Field (T)')
plt.title('Magnetic Field in Solenoid vs. Number of Turns')
plt.legend()
plt.grid(True)
plt.show()
