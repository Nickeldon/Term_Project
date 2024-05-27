import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

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

start_time = 0

class setInterval:  # I use JS and like the setInterval function
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

# Calculate the memory consumption in a different interval that iterates more slowly for more performance
mem_consump = 0
thread_interv = None

# Source from Charrat Mohamed's Projectile Motion Assignment
# Function to calculate the memory used by the simulation
def get_Mem_Usage():
    global mem_consump  # According to Stack Overflow [To update a global variable in a local scoped function]
    pr = psutil.Process(os.getpid())
    mem_consump = round(pr.memory_info()[0] / float(2 ** 20), 2)
    return mem_consump

if used_mem_lib:
    thread_interv = setInterval(get_Mem_Usage, 1000)
    thread_interv.init()

# Constants for loading bar
bar_len = 50
update_interval = 100

# The percentage tolerance for the ideal average Magnetic field value
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
            print('The Timer has not been started!', end='\r')

timer = Timer()

timer.start_timer()

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

class Solenoid:
    def __init__(self, I, L, R):
        self.I = I  # Current of the solenoid (constant)
        self.L = L  # Length of the Solenoid (constant)
        self.R = R
        self.mu_0 = 4 * np.pi * 1e-7  # mu_0
        self.tol = tol_percent  # The tolerance of our ideal Magnetic field value
        self.B_avg_cache = {}  # Keeps in cache the previously calculated Magnetic field values for performance purposes

    def _get_B_center(self, N):
        n = N / self.L
        B_cent = self.mu_0 * n * self.I
        return B_cent

    def _get_a0(self, z):
        # Placeholder for a0(z) based on Bz(0, 0, z)
        return self._get_B_center(N=1)

    def _get_B_z(self, rho, z, N):
        # McDonald model series expansion for B_z(rho, z)
        a0 = self._get_a0(z)
        Bz = 0
        for n in range(8):  # Sum up to 7th order
            a0_n = np.polyder(np.poly1d([a0]), n)
            Bz += ((-1) ** n * a0_n(z) / (math.factorial(n) ** 2) * (rho / 2) ** (2 * n))
        return Bz

    def _get_B_rho(self, rho, z, N):
        # McDonald model series expansion for B_rho(rho, z)
        a0 = self._get_a0(z)
        Brho = 0
        for n in range(8):  # Sum up to 7th order
            a0_n = np.polyder(np.poly1d([a0]), n)
            Brho += ((-1) ** (n + 1) * a0_n(z) / (math.factorial(n) * math.factorial(n + 1)) * (rho / 2) ** (2 * n + 1))
        return Brho

    def _get_avg_B(self, N, iterations=None, showProg=False, skip_bar=False):
        if N in self.B_avg_cache:  # Check if the result is already in the cache
            return self.B_avg_cache[N]

        z_pts = np.linspace(-self.L / 2, self.L / 2, 10000)
        B_pts = []
        for i, val in enumerate(z_pts):
            B_pts.append(self._get_B_z(0, val, N))
            if not skip_bar and (i % update_interval == 0 or i == len(z_pts) - 1):
                if not iterations:
                    iter_arr = ('', '')
                else:
                    iter_arr = iterations
                print_progress(i + 1, len(z_pts), head='Calculating B:', asyncIter=iter_arr)
        B_avg = np.mean(B_pts)
        B_avg_N.append(B_avg)
        self.B_avg_cache[N] = B_avg  # Store the result in the cache
        return B_avg

    def _verif_if_within_tol(self, N):
        print(f'Initializing simulation... elapsed time: {timer.get_timer()}s', end='\r')
        B_cent = self._get_B_center(N)
        B_avg = self._get_avg_B(N, skip_bar=True)
        return abs(B_avg - 9) / 9 <= self.tol

class GoldenSectionSearch:
    def __init__(self, f, a, b, tol=1e-5):
        self.f = f
        self.a = a
        self.b = b
        self.tol = tol
        self.gr = (np.sqrt(5) + 1) / 2

    def estim_max_iter(self):
        return math.ceil(math.log(self.tol / abs(self.b - self.a)) / math.log(1 / self.gr))

    def search(self):
        es_max = self.estim_max_iter()  # Estimated maximum number of iterations through the golden section search
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
R = 0.1

# Create Solenoid instance
solenoid = Solenoid(I, L, R)

def objective_N(N, iter_state):
    B_c = solenoid._get_B_center(N)
    B_avg_val = solenoid._get_avg_B(N, iterations=iter_state, skip_bar=False, showProg=True)
    B_avg_N.append(B_avg_val)
    return abs(B_avg_val - 9)

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

# Clear console
os.system('cls')

# Display results
print(f"""
Finished simulation in: {timer.get_timer()}s
Optimal N: {round(optimal_N)}
Magnetic Field Value of optimal N: {B_avg_N[-1]} T
Amount of Iterations: {i}
Percentage Error: {round(abs(B_avg_N[-1] - 9) / 9 * 100, 2)}%
Memory used: {mem_consump} MB

Note:
I would like to thank Prof. Kirk T. McDonald of Princeton University for the work on 'On the Magnetic Field of a Finite Solenoid' and Prof. Charrat Mohamed for the CS50P Introduction course by HarvardX.""")

if used_mem_lib:
    thread_interv.abort()
