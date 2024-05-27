import numpy as np
import psutil
import os
import time
import math

def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2

class Timer:
    def __init__(self):
        self.t_0 = 0  # Initial value of the timer
        self.started = False  # If timer is started or not.
        
    def start_timer(self):
        self.t_0 = time.time()  # Set the initial time to the current time
        self.started = True
    
    def get_timer(self):
        elapsed = time.time() - self.t_0
        if self.started:
            return int(elapsed)
        else:
            return 0

timer = Timer()
timer.start_timer()

bar_len = 50  # Length of the progress bar
update_interval = 100

def print_progress(iter, max_iter, head, asyncIter = ['', '']):
    percent = int(100 * (iter / int(max_iter)))
    
    disp_percent = percent if percent >= 10 else '0' + str(percent)
    
    blocks = int((bar_len * iter / int(max_iter)))
    bar = '█' * blocks + '-' * (bar_len - blocks)
    asyncIter_arr = asyncIter
        
    mem_consump = memory_usage_psutil()
    print(f'{head} |{bar}| {disp_percent}% ({iter}/{max_iter}) | ({asyncIter_arr[0]} / {asyncIter_arr[1]}) | Mem usage: {mem_consump:.2f} MB | elapsed_time: {timer.get_timer()}s  ', end='\r')
    if iter == max_iter:
        print('\r', end='\r')

class solenoid:
    def __init__(self, I, R1, R2, L, rho, n_terms=5):
        self.I = I
        self.R1 = R1
        self.R2 = R2
        self.L = L
        self.rho = rho
        self.n_terms = n_terms
        self.mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)

    # On-axis field Bz(0, 0, z)
    def _verif_B_on_axis(self, N, z):
        z1 = -self.L / 2
        z2 = self.L / 2
        term1 = (z - z1) * np.log((np.sqrt(self.R2**2 + (z - z1)**2) + self.R2) / (np.sqrt(self.R1**2 + (z - z1)**2) + self.R1))
        term2 = (z - z2) * np.log((np.sqrt(self.R2**2 + (z - z2)**2) + self.R2) / (np.sqrt(self.R1**2 + (z - z2)**2) + self.R1))
        return (self.mu0 * self.I * N / (2 * self.L * (self.R2 - self.R1))) * (term1 - term2)

    # Derivatives of Bz(0, 0, z)
    def _get_B_z(self, z, n, N):
        if n == 0:
            return self._verif_B_on_axis(N, z)
        else:
            z1 = -self.L / 2
            z2 = self.L / 2
            term1 = ((z - z1)**n / np.sqrt(self.R2**2 + (z - z1)**2)**(n+1) - (z - z2)**n / np.sqrt(self.R2**2 + (z - z2)**2)**(n+1))
            term2 = ((z - z1)**n / np.sqrt(self.R1**2 + (z - z1)**2)**(n+1) - (z - z2)**n / np.sqrt(self.R1**2 + (z - z2)**2)**(n+1))
            return (self.mu0 * self.I * N / (2 * self.L * (self.R2 - self.R1))) * (term1 - term2)

    def _get_B_field(self, N, z):

        #According to the Mcdonald's model
        
        # Calculate Bz and Brho using series expansion
        Bz = 0
        Brho = 0

        for n in range(self.n_terms + 1):
            factor = (-1)**n / (math.factorial(n)**2)
            Bz += factor * (self.rho**(2*n)) * self._get_B_z(z, 2*n, N)
            if n > 0:
                Brho += factor * (self.rho**(2*n-1)) * self._get_B_z(z, 2*n-1, N) / n

        return Brho, Bz

    def find_optimal_turns(self, target_B, tolerance):
        N_min, N_max = 1, 100000  # Set an initial search range for the number of turns
        z_points = np.linspace(-self.L / 2, self.L / 2, 100)  # Sample points along the solenoid axis

        while N_min < N_max:
            N = (N_min + N_max) // 2
            Bz_values = np.array([self._get_B_field(N, z)[1] for z in z_points])
            B_avg = np.mean(Bz_values)

            if abs(B_avg - target_B) <= tolerance:
                return N, B_avg
            elif B_avg < target_B:
                N_min = N + 1
            else:
                N_max = N - 1

        return N, B_avg
    
# Example usage
I = 75  # Current in the solenoid (A)
R1 = 0.04125  # Inner radius of the solenoid (m)
R2 = 0.04637  # Outer radius of the solenoid (m)
L = 0.500  # Length of the solenoid (m)
rho = 0.01  # Radial position (m)
n_terms = 5  # Number of terms in the series expansion
target_B = 9  # Target average magnetic field (T)
tolerance = 0.45  # Allowed deviation from the target field (T)

getMagSolenoid = solenoid(I, R1, R2, L, rho, n_terms)
N_optimal, B_avg = getMagSolenoid.find_optimal_turns(target_B, tolerance)

print(f"Optimal number of turns: {N_optimal}, Achieved average B: {B_avg:.2f} T")