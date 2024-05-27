import numpy as np
import psutil
import os
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

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

################################################################################

class Solenoid:
    def __init__(self, I, R, L, rho, n_terms=5):
        self.I = I
        self.R1 = R[0]
        self.R2 = R[1]
        self.L = L
        self.rho = rho
        self.n_tot = n_terms
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
        # Calculate Bz and Brho using series expansion
        Bz = 0
        Brho = 0

        for n in range(self.n_tot + 1):
            factor = (-1)**n / (math.factorial(n)**2)
            Bz += factor * (self.rho**(2*n)) * self._get_B_z(z, 2*n, N)
            if n > 0:
                Brho += factor * (self.rho**(2*n-1)) * self._get_B_z(z, 2*n-1, N) / n

        return Brho, Bz

    def find_optimal_turns(self, B_target, tol):
        N_min, N_max = 1, 100000  # Set an initial search range for the number of turns
        z_points = np.linspace(-self.L / 2, self.L / 2, 1000)  # Sample points along the solenoid axis

        while N_min < N_max:
            N = (N_min + N_max) // 2
            Bz_values = np.array([self._get_B_field(N, z)[1] for z in z_points])
            B_avg = np.mean(Bz_values)

            print_progress(N - N_min, N_max - N_min, "Finding optimal number of turns")

            if abs(B_avg - B_target) <= tol:
                return N, B_avg
            elif B_avg < B_target:
                N_min = N + 1
            else:
                N_max = N - 1

        return N, B_avg

    def get_avg_field(self, N):
        z_points = np.linspace(-self.L / 2, self.L / 2, 100)
        Bz_values = np.array([self._get_B_field(N, z)[1] for z in z_points])
        B_avg = np.mean(Bz_values)
        return B_avg

#Class for golden section search
class golden_section_search:
    def __init__(self, I, R, L, rho, n_terms, B_target, tol):
        self.I = I # Current
        self.R1 = R[0] # Inner radius
        self.R2 = R[1]  # Outer radius
        self.L = L  # Length
        self.rho = rho  # Radial distance
        self.n_max = n_terms
        self.B_target = B_target    # Target magnetic field
        self.tol = tol  # tolerance for the average magnetic field
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.solenoid = Solenoid(self.I, (self.R1, self.R2), self.L, self.rho, self.n_max)

    def estim_max_iter(self):
        return math.ceil(math.log(self.tol/abs(self.B_target - 0)) / math.log(1/self.phi))

    # Start the golden section search algorithm
    def search(self):
        
        max_iter = self.estim_max_iter()
        
        # Define the golden ratio and its inverse
        invphi = 1 / self.phi

        # Define the initial search range for the number of turns
        a, b = 1000, 100000
        h = b - a
        # Define two points within the search range using the golden ratio
        c = b - invphi * h
        d = a + invphi * h
        # Calculate the absolute difference between the average field at points c and d and the target field
        fc = abs(self.solenoid.get_avg_field(c) - self.B_target)
        fd = abs(self.solenoid.get_avg_field(d) - self.B_target)

        # Continue the loop until the difference between the upper and lower bounds is less than the tol
        i = 0
        while h > self.tol:
            i += 1
            #print(i, max_iter)
            # If the absolute difference at point c is less than at point d, update the upper bound and corresponding values
            if fc < fd:
                b = d
                d = c
                fd = fc
                h = b - a
                c = b - invphi * h
                fc = abs(self.solenoid.get_avg_field(c) - self.B_target)
            else:
                # If the absolute difference at point d is less than at point c, update the lower bound and corresponding values
                a = c
                c = d
                fc = fd
                h = b - a
                d = a + invphi * h
                fd = abs(self.solenoid.get_avg_field(d) - self.B_target)

            # Print the progress of the search
            print_progress(iter = i, max_iter = max_iter * 4, head="Optimize with ggs")


        if fc < fd:
            optimal_N = c
            optimal_B_avg = self.solenoid.get_avg_field(optimal_N)
        else:
            optimal_N = d
            optimal_B_avg = self.solenoid.get_avg_field(optimal_N)

        return optimal_N, optimal_B_avg

def plot_B_vs_turns():
    
    ###            ###
    ### Parameters ###
    ###            ###
    I = 75 # Current (A)
    R1 = 0.04125 # Inner radius (m)
    R2 = 0.04637 # Outer radius (m)
    L = 0.500 # Length (m)
    rho = 0.01 # Radial distance (m)
    n_terms = 5 
    B_target = 9 # Target magnetic field (T)
    tol = 0.1 # Tolerance for the average magnetic field (1/100 %)

    optimal_N, optimal_B_avg = golden_section_search(I, (R1, R2), L, rho, n_terms, B_target, tol).search()

    os.system("cls")

    print(f"Optimal number of turns: {optimal_N}, Achieved average B: {optimal_B_avg:.2f} T")

    N_values = np.arange(10000, 60000, 1000)  # Adjusted to range in the 10,000s
    B_avg_values = []

    solenoid = Solenoid(I, (R1, R2), L, rho, n_terms)

    final_arr = []

    for N in N_values:
        B_avg = solenoid.get_avg_field(N)
        B_avg_values.append(B_avg)
        percent_diff = (abs(B_avg - B_target))/((B_avg + B_target) / 2)
        #final_dict[N] = [round(B_avg, 2), round(percent_diff, 2)]
        final_arr.append([N, round(B_avg, 2), round(percent_diff, 2) if round(percent_diff, 2) > 0.01 else f'  > {round(percent_diff, 3)} <'])
        print_progress(N - 10000, 50000, "Calculating average magnetic field")

    os.system("cls")

    print('_____________________Results______________________')    
    print(f"""
    ############################################
    > Optimal number of turns: {optimal_N}
    > Achieved avg B: {optimal_B_avg:.2f} T
    ############################################
          """)
    
    #I love pandas ;D
    df = pd.DataFrame(final_arr, columns=['Number of Turns', 'Average B (T)', 'Percentage difference (%)'])
    df.set_index('Number of Turns', inplace=True)

    print(df)

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, B_avg_values, label='Average Magnetic Field')
    plt.axhline(y=B_target, color='r', linestyle='--', label='Target B = 9T')
    plt.axvline(x=48000, color='b', linestyle='--', label='Center = 48000 Coils')
    plt.axvline(x=48000 + 4500, color='g', linestyle='--', label='Upper tol = 52500 Coils')
    plt.axvline(x=48000 - 4500, color='g', linestyle='--', label='Lower tol = 43500 Coils')
    plt.ylim(0, 20)  # Set y-axis range from 0T to 20T
    plt.xlim(1, 100000)  # Set x-axis range from 1 to 100000 coils

    plt.xlabel('Number of Turns')
    plt.ylabel('Average Magnetic Field (T)')
    plt.title('Average Magnetic Field vs. Number of Turns')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
plot_B_vs_turns()