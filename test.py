import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

used_mem_lib = True

try:
    import psutil
    import threading
except:
    used_mem_lib = False
    print("""
    ###                                                                            ###
    ### Please use the psutil and threading modules to show the memory consumption ###
    ###                                                                            ###""")

start_time = 0
class setInterval: #I use JS and like the setInterval function
    def __init__(self, func, ms):
        self.func = func
        self.ms = ms
        self.thread = threading.Timer(self.ms / 1000, self.func_wrapper)
        
    def func_wrapper(self):
        setInterval(self.func, self.ms).init()
        self.func()
    
    def init(self):    
        # If I don't use another thread, it will just stay stuck in this function as it will loop indefenitely
        self.thread.start()
    
    def abort(self):
        self.thread.cancel()
        

# Calculate the memory consumption in a different interval that iterates more slowly for more performance
mem_consump = 0
thread_interv = None

#Source from Charrat Mohamed's Projectile Motion Assignment
#Function to calculate the memory used by the simulation
def get_Mem_Usage():
    global mem_consump #According to Stack Overflow [To update a global variable in a local scoped function]
    pr = psutil.Process(os.getpid())
    
    # According to https://pythonhow.com/how/limit-floats-to-two-decimal-points/ [To limit floats to two decimal points]
    
    mem_consump = round(pr.memory_info()[0] / float(2 ** 20), 2)
    return mem_consump

if used_mem_lib:
    thread_interv = setInterval(get_Mem_Usage, 1000)
    thread_interv.init()

# Constants for loading bar
bar_len = 50
update_interval = 100

#The percentage tolerance for the ideal average Magnetic field value
tol_percent = 0.075

B_avg_N = []

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
            for _ in range(0, 10):
                print('The Timer has not been started!', end='\r')

timer = Timer()

timer.start_timer()

def print_progress(iter, max_iter, head, asyncIter = ['', '']):
    percent = int(100 * (iter / int(max_iter)))
    blocks = int((bar_len * iter / int(max_iter)))
    bar = '█' * blocks + '-' * (bar_len - blocks)
    asyncIter_arr = asyncIter
        
    print(f'{head} |{bar}| {percent}% ({iter}/{max_iter}) | ({asyncIter_arr[0]} / {asyncIter_arr[1]}) | Mem usage: {mem_consump} MB | elapsed_time: {timer.get_timer()}s  ', end='\r\r')
    print('\r', end='\r')
    if iter == max_iter:
        print('\r', end='\r')

class Solenoid:
    def __init__(self, I, L):
        self.I = I
        self.L = L
        self.mu_0 = 4 * np.pi * 1e-7
        self.tol = tol_percent
        self.dl = 1  # Small length element for integration
        self.B_0= 0
        self.B_avg_cache = {}

    def _get_B_center(self, N):
        n = N / self.L
        B_cent = self.mu_0 * n * self.I
        return B_cent

    def _get_B_z(self, z, N):
            
        n = N / self.L  # Coil density
        for s in np.arange(-self.L/2, self.L/2, self.dl):
            r = np.sqrt(z**2 + (s - z)**2)
            dB = (self.mu_0 * n * self.I * self.dl) / (2 * np.pi * r)
            self.B_0 += dB
        return self.B_0

    def _get_avg_B(self, N, iterations = None, showProg = False, skip_bar=False):
        if N in self.B_avg_cache:  # Check if the result is already in the cache
            return self.B_avg_cache[N]

        z_pts = np.linspace(-self.L / 2, self.L / 2, 100000)
        B_pts = []
        for i, val in enumerate(z_pts):
            B_pts.append(self._get_B_z(val, N))
            if not skip_bar and (i % update_interval == 0 or i == len(z_pts) - 1):
                if not iterations:
                    iter_arr = ('', '')
                else:
                    iter_arr = iterations
                print_progress(i + 1, len(z_pts), head='Calculating B:', asyncIter = iter_arr)
        B_avg = np.mean(B_pts)
        self.B_avg_cache[N] = B_avg  # Store the result in the cache
        return B_avg

    def _verif_if_within_tol(self, N):
        print(f'Initializing simulation... ** Note: The initialization will take a long time if the iteration value is high **', end='\r')
        B_cent = self._get_B_center(N)
        B_avg = self._get_avg_B(N, skip_bar=True)
        return abs(B_avg - B_cent) / B_cent <= self.tol

class GoldenSectionSearch:
    def __init__(self, f, a, b, tol=1e-5):
        self.f = f
        self.a = a
        self.b = b
        self.tol = tol
        self.gr = (np.sqrt(5) + 1) / 2
    
    def estim_max_iter(self):
        return math.ceil(math.log(self.tol/abs(self.b-self.a)) / math.log(1/self.gr))

    def search(self):
        es_max = self.estim_max_iter() # Estimated maximum number of iterations through the golden section search
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
I = 75  # Current in the solenoid (Will not change as is irrelevant to experiment) (A)
L = 0.5  # Length of the solenoid (m)

# Create Solenoid instance
solenoid = Solenoid(I, L)

def objective_N(N, iter_state):
    B_c = solenoid._get_B_center(N)
    B_avg_val = solenoid._get_avg_B(N, iterations = iter_state, skip_bar=False, showProg=True)
    B_avg_N.append(B_avg_val)
    return abs(B_avg_val - B_c) / B_c

def find_initial_bounds():
    N = 1
    while not solenoid._verif_if_within_tol(N):
        N *= 2
        if N > 1e5:  # If N becomes too large, break
            break
    min = N // 2 if N < 1e5 else N // 4
    max = N if N < 1e5 else N // 2
    return min, max

# Clear console
os.system('cls')

a_N, b_N = find_initial_bounds()

gss = GoldenSectionSearch(objective_N, a_N, b_N)

# Optimize for the minimum number of turns
optimal_N, i = gss.search()

# Print the optimal number of coils
print(f"The optimal number of coils (N) for a uniform magnetic field within {tol_percent * 100}% tolerance is: {optimal_N}")

# Calculate the magnetic fields for plotting
N_arr = np.linspace(a_N, b_N, 2 * i)
B_avg_arr = []
print('The graph is being drawn. Please wait')
for i, N in enumerate(N_arr):
    B_avg_arr.append(solenoid._get_avg_B(N, showProg = True, skip_bar=True))
    print_progress(i + 1, len(N_arr), head='Calculating B_avg for plot:')
B_cent_arr = [solenoid._get_B_center(N) for N in N_arr]

# Calculate the 5% tolerance bands
B_mid_opt = solenoid._get_B_center(optimal_N)
tol_band_upper = B_mid_opt * (1 + solenoid.tol)
tol_band_lower = B_mid_opt * (1 - solenoid.tol)

if used_mem_lib:
    thread_interv.abort()

# Plot the results
plt.figure(figsize=(10, 6))
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
