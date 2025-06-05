import numpy as np
from celluloid import Camera
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
from tqdm import tqdm, trange
import imageio.v2 as imageio
import time
import copy
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse
import json
import os
from io import BytesIO
import string

class PolymerLatticeSimulation:
    """
    A class for simulating polymer chains on a 2D lattice using Monte Carlo methods.
    """
    
    def __init__(self, l_size):
        """
        Initialize the lattice.
        
        Args:
            l_size (int): Size of the square lattice (l_size x l_size)
        """
        self.lat = np.array([[None for i in range(l_size)] for j in range(l_size)])
        self.l_size = l_size
        self.poly = None
        self.coords = None
        self.energy = 0
        self.vlat = None
        
    def embed(self, p_n, poly):
        """
        Embed polymer chains randomly on the lattice.
        
        Args:
            p_n (int): Number of polymer chains to embed
            poly (str): String representing the polymer sequence (e.g., 'abcd')
        """
        self.poly = poly
        p_size = len(poly)
        l_size = self.l_size
        lat = self.lat
        coords = []
        i = 1
        
        while i <= p_n:
            r_pick = [[1,0], [-1,0], [0,1], [0,-1]]
            x = np.random.randint(l_size)
            y = np.random.randint(l_size)
            
            if lat[x][y] == None:
                i += 1
                pos = [[x, y]]
                lat[x][y] = poly[0] + str(i-1)
                j = 1
                
                while j <= p_size - 1:
                    x = pos[-1][0]
                    y = pos[-1][1]
                    r = np.random.randint(4)
                    x += r_pick[r][0]
                    y += r_pick[r][1]
                    
                    if 0 <= x < l_size and 0 <= y < l_size and lat[x][y] == None:
                        j += 1
                        lat[x][y] = poly[j-1] + str(i-1)
                        pos.append([x, y])
                        
                coords.append(pos)
        
        self.coords = coords
        self.lat = lat
        n = len(lat)
        vlat = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if lat[i][j] != None:
                    vlat[i][j] = int(lat[i][j][1])
                else: 
                    vlat[i][j] = 0
                    
        self.vlat = vlat
        return None
    
    def latenergy(self, energy):
        """
        Calculate the total energy of the current lattice configuration.
        
        Args:
            energy (np.array): Energy matrix for interactions between different monomers
            
        Returns:
            float: Total energy of the system
        """
        lat = self.lat
        poly = self.poly
        the = 0
        n = len(lat)
        coords = self.coords
        
        for e in coords:
            for g in e:
                i = g[0]
                j = g[1]
                if lat[i][j] != None:
                    val = lat[i][j][0]
                    index = lat[i][j][1]
                    
                    # Check all four neighbors
                    neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                    
                    for ni, nj in neighbors:
                        if (0 <= ni < n and 0 <= nj < n and 
                            lat[ni][nj] != None and lat[ni][nj][1] != index):
                            the += energy[poly.index(val)][poly.index(lat[ni][nj][0])]
                            
        self.energy = the / 2
        return the / 2

    def randompick(self, opt=None):
        """
        Randomly pick a monomer position from the lattice.
        
        Args:
            opt (int, optional): Specific polymer chain to pick from
            
        Returns:
            list: [x, y] coordinates of the selected monomer
        """
        coords = self.coords
        if opt != None:
            return coords[opt][np.random.randint(len(coords[0]))]
        g = coords[np.random.randint(len(coords))]
        return g[np.random.randint(len(g))]
    
    def avmoves(self, pos):
        """
        Find all available moves for a monomer at given position.
        
        Args:
            pos (list): [x, y] position of the monomer
            
        Returns:
            list: List of available positions [[x1, y1], [x2, y2], ...]
        """
        lat = self.lat
        poly = self.poly
        n = len(lat)
        o = lat[pos[0]][pos[1]]
        val = o[0]
        index = int(o[1])
        avpos = []
        
        # Calculate neighborhood bounds
        cx = -1
        cy = -1
        if pos[0] == 0: 
            u = pos[0]
            cx = 0
        else: 
            u = pos[0] - 1
        if pos[0] == n - 1: 
            d = pos[0] 
        else: 
            d = pos[0] + 1
        if pos[1] == 0: 
            l = pos[1]
            cy = 0
        else: 
            l = pos[1] - 1
        if pos[1] == n - 1: 
            r = pos[1] 
        else: 
            r = pos[1] + 1
            
        neig = lat[u:d+1, l:r+1]
        axes1 = [[1,1], [1,-1], [-1,1], [-1,-1]]
        axes2 = [[1,1], [1,-1], [-1,1], [-1,-1], [2,0], [-2,0], [0,2], [0,-2]]
        
        # Find position of this monomer in the polymer sequence
        monomer_pos_in_chain = poly.index(val)
        
        # Handle different cases based on monomer position in polymer
        if monomer_pos_in_chain == 0:  # First monomer
            neig1 = [int(np.where(neig==(poly[1]+str(index)))[0][0])+pos[0]+cx,
                    int(np.where(neig==(poly[1]+str(index)))[1][0])+pos[1]+cy]
            for i in axes2:
                x = pos[0] + i[0]
                y = pos[1] + i[1]
                if (0 <= x < n and 0 <= y < n and lat[x][y] == None and 
                    (int(abs(x-neig1[0])+abs(y-neig1[1])) == 1)):
                    avpos.append([x, y])
        elif monomer_pos_in_chain == len(poly) - 1:  # Last monomer
            neig1 = [int(np.where(neig==(poly[-2]+str(index)))[0][0])+pos[0]+cx,
                    int(np.where(neig==(poly[-2]+str(index)))[1][0])+pos[1]+cy]
            for i in axes2:
                x = pos[0] + i[0]
                y = pos[1] + i[1]
                if (0 <= x < n and 0 <= y < n and lat[x][y] == None and 
                    (int((abs(x-neig1[0])+abs(y-neig1[1])))==1)):
                    avpos.append([x, y])
        else:  # Middle monomer
            neig1 = [int(np.where(neig==(poly[monomer_pos_in_chain-1]+str(index)))[0][0])+pos[0]+cx,
                    int(np.where(neig==(poly[monomer_pos_in_chain-1]+str(index)))[1][0])+pos[1]+cy]
            neig2 = [int(np.where(neig==(poly[monomer_pos_in_chain+1]+str(index)))[0][0])+pos[0]+cx,
                    int(np.where(neig==(poly[monomer_pos_in_chain+1]+str(index)))[1][0])+pos[1]+cy]
            for i in axes1:
                x = pos[0] + i[0]
                y = pos[1] + i[1]
                if (0 <= x < n and 0 <= y < n and lat[x][y] == None and 
                    (int(abs(x-neig1[0])+abs(y-neig1[1])) == 1) and 
                    (int(abs(x-neig2[0])+abs(y-neig2[1])) == 1)):
                    avpos.append([x, y])
                    
        return avpos
    
    def move(self, ipos, fpos):
        """
        Move a monomer from initial position to final position.
        
        Args:
            ipos (list): Initial position [x, y]
            fpos (list): Final position [x, y]
        """
        lat = self.lat
        poly = self.poly
        nlat = lat.copy()
        val = nlat[ipos[0]][ipos[1]][0]
        index = int(nlat[ipos[0]][ipos[1]][1])
        coords = self.coords
        ncoords = coords.copy()
        ncoords[index-1][poly.index(val)] = fpos
        self.coords = ncoords
        vlat = self.vlat
        nvlat = vlat.copy()
        nlat[fpos[0]][fpos[1]] = nlat[ipos[0]][ipos[1]]
        nvlat[fpos[0]][fpos[1]] = nvlat[ipos[0]][ipos[1]]
        nlat[ipos[0]][ipos[1]] = None
        nvlat[ipos[0]][ipos[1]] = 0
        self.lat = nlat
        self.vlat = nvlat
        return None

    def visual(self):
        """
        Return the visual representation of the lattice.
        
        Returns:
            np.array: 2D array for visualization
        """
        return self.vlat


class SimulationConfig:
    """
    Configuration class for simulation parameters.
    """
    
    def __init__(self):
        self.l_size = 20
        self.p_n = 6
        self.polymer_length = 4  # Changed from polymer_sequence
        self.polymer_sequence = 'abcd'  # Auto-generated based on length
        self.n_sim = 1000
        self.interaction_energy = -1
        self.energy_mode = 'same'  # 'same' or 'custom'
        self.custom_energies = {}  # For storing individual monomer energies
        self.sampling_rate = 10
        self.n_runs = 1
        self.output_dir = 'simulation_output'
        self.kt = 1.0
        
    def generate_polymer_sequence(self):
        """Generate polymer sequence based on length."""
        if self.polymer_length > 26:
            raise ValueError("Polymer length cannot exceed 26 (alphabet limit)")
        self.polymer_sequence = string.ascii_lowercase[:self.polymer_length]
        
    def setup_energy_matrix(self):
        """Setup energy matrix based on configuration."""
        poly = self.polymer_sequence
        energy = np.zeros((len(poly), len(poly)))
        
        if self.energy_mode == 'same':
            # Set all diagonal elements to the same value
            for i in range(len(poly)):
                energy[i][i] = self.interaction_energy
        else:  # custom mode
            # Set individual energies for each monomer type
            for i, monomer in enumerate(poly):
                if monomer in self.custom_energies:
                    energy[i][i] = self.custom_energies[monomer]
                else:
                    energy[i][i] = self.interaction_energy  # fallback
                    
        return energy
        
    def load_from_file(self, filename):
        """Load configuration from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        # Regenerate polymer sequence after loading
        self.generate_polymer_sequence()
                    
    def save_to_file(self, filename):
        """Save configuration to JSON file."""
        data = {attr: getattr(self, attr) for attr in dir(self) 
                if not attr.startswith('_') and not callable(getattr(self, attr))}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def interactive_setup(self):
        """Interactive setup for simulation parameters."""
        print("\n=== Polymer Lattice Simulation Setup ===")
        print("Press Enter to use default values shown in brackets\n")
        
        # Get lattice size
        try:
            response = input(f'Lattice size [{self.l_size}]: ').strip()
            if response:
                self.l_size = int(response)
        except ValueError:
            print("Invalid input, using default value")
            
        # Get number of polymers
        try:
            response = input(f'Number of polymer chains [{self.p_n}]: ').strip()
            if response:
                self.p_n = int(response)
        except ValueError:
            print("Invalid input, using default value")
            
        # Get polymer length
        try:
            response = input(f'Polymer length [{self.polymer_length}]: ').strip()
            if response:
                self.polymer_length = int(response)
                if self.polymer_length > 26:
                    print("Warning: Polymer length limited to 26. Setting to 26.")
                    self.polymer_length = 26
        except ValueError:
            print("Invalid input, using default value")
            
        # Generate polymer sequence
        self.generate_polymer_sequence()
        print(f'Generated polymer sequence: {self.polymer_sequence}')
        
        # Get number of iterations
        try:
            response = input(f'Number of iterations [{self.n_sim}]: ').strip()
            if response:
                self.n_sim = int(response)
        except ValueError:
            print("Invalid input, using default value")
            
        # Get energy configuration mode
        print(f'\nEnergy configuration:')
        print(f'1. Same energy for all monomers')
        print(f'2. Custom energy for each monomer type')
        try:
            response = input(f'Choose mode (1/2) [1]: ').strip()
            if response == '2':
                self.energy_mode = 'custom'
                self.custom_energies = {}
                print(f'\nEnter interaction energy for each monomer type:')
                for monomer in self.polymer_sequence:
                    try:
                        energy_val = input(f'Energy for monomer "{monomer}" [{self.interaction_energy}]: ').strip()
                        if energy_val:
                            self.custom_energies[monomer] = float(energy_val)
                        else:
                            self.custom_energies[monomer] = self.interaction_energy
                    except ValueError:
                        print(f"Invalid input for {monomer}, using default value")
                        self.custom_energies[monomer] = self.interaction_energy
            else:
                self.energy_mode = 'same'
                # Get interaction energy for same mode
                try:
                    response = input(f'Interaction energy [{self.interaction_energy}]: ').strip()
                    if response:
                        self.interaction_energy = float(response)
                except ValueError:
                    print("Invalid input, using default value")
        except ValueError:
            print("Invalid input, using same energy mode")
            self.energy_mode = 'same'
            
        # Get sampling rate
        try:
            response = input(f'Sampling rate (save every Nth frame) [{self.sampling_rate}]: ').strip()
            if response:
                self.sampling_rate = int(response)
        except ValueError:
            print("Invalid input, using default value")
            
        # Get number of runs
        try:
            response = input(f'Number of simulation runs [{self.n_runs}]: ').strip()
            if response:
                self.n_runs = int(response)
        except ValueError:
            print("Invalid input, using default value")
            
        print(f"\nConfiguration summary:")
        print(f"Lattice size: {self.l_size}x{self.l_size}")
        print(f"Number of polymers: {self.p_n}")
        print(f"Polymer length: {self.polymer_length}")
        print(f"Polymer sequence: {self.polymer_sequence}")
        print(f"Iterations: {self.n_sim}")
        print(f"Energy mode: {self.energy_mode}")
        if self.energy_mode == 'same':
            print(f"Interaction energy: {self.interaction_energy}")
        else:
            print("Custom energies:")
            for monomer, energy in self.custom_energies.items():
                print(f"  {monomer}: {energy}")
        print(f"Sampling rate: {self.sampling_rate}")
        print(f"Number of runs: {self.n_runs}")


def run_simulation(config):
    """
    Run the polymer lattice simulation with given configuration.
    
    Args:
        config (SimulationConfig): Configuration object
    """
    # Create output directory
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # Save configuration
    config.save_to_file(os.path.join(config.output_dir, 'config.json'))
    
    # Generate polymer sequence and setup energy matrix
    config.generate_polymer_sequence()
    poly = config.polymer_sequence
    p_n = config.p_n
    l_size = config.l_size
    n_sim = config.n_sim
    kt = config.kt
    
    # Setup energy matrix
    energy = config.setup_energy_matrix()
    
    # Generate colors for visualization
    colors_list = ['red','blue','green','orange','purple','brown','pink','gray','olive','cyan']
    
    sample = max(1, int(n_sim / config.sampling_rate))
    
    print(f"\nStarting simulation with {config.n_runs} runs...")
    print(f"Each run: {n_sim} iterations, sampling every {sample} steps")
    print(f"Polymer sequence: {poly}")
    
    start_time = time.time()
    
    for run_count in range(1, config.n_runs + 1):
        print(f"\n--- Run {run_count}/{config.n_runs} ---")
        
        # Initialize lattice
        lat = PolymerLatticeSimulation(l_size)
        lat.embed(p_n, poly)
        
        # Storage for this run
        lats = []
        flats = []
        energies = []
        emin = 1e5
        statemin = []
        fstatemin = []
        
        filename = f'run_{run_count}_length_{config.polymer_length}_energy_{config.energy_mode}'
        
        # Progress bar for this run
        pbar = tqdm(total=n_sim, desc=f"Run {run_count}", position=0, leave=True)
        
        i = 0
        while i < n_sim:
            pos = lat.randompick()
            moves = lat.avmoves(pos)
            
            if len(moves) == 0:
                i += 1
                if i % sample == 0 or i == n_sim:
                    lats.append(lat.visual())
                    flats.append(copy.deepcopy(lat.coords))
                energies.append(lat.latenergy(energy))
                pbar.update(1)
                continue
                
            # Monte Carlo move
            pick = moves[np.random.randint(len(moves))]
            ei = lat.latenergy(energy)
            lat.move(pos, pick)
            ei1 = lat.latenergy(energy)
            w = np.exp(-(ei1 - ei) / kt)
            
            if w >= 1:
                # Accept move
                i += 1
                if i % sample == 0 or i == n_sim:
                    lats.append(lat.visual())
                    flats.append(copy.deepcopy(lat.coords))
                te = lat.latenergy(energy)
                if te < emin:
                    emin = te
                    statemin = lat.visual()
                    fstatemin = copy.deepcopy(lat.coords)
                energies.append(te)
                pbar.update(1)
            elif w > np.random.rand():
                # Accept move with probability w
                i += 1
                if i % sample == 0 or i == n_sim:
                    lats.append(lat.visual())
                    flats.append(copy.deepcopy(lat.coords))
                energies.append(lat.latenergy(energy))
                pbar.update(1)
            else:
                # Reject move
                i += 1
                lat.move(pick, pos)  # Move back
                if i % sample == 0 or i == n_sim:
                    lats.append(lat.visual())
                    flats.append(copy.deepcopy(lat.coords))
                energies.append(lat.latenergy(energy))
                pbar.update(1)
        
        pbar.close()
        
        # Generate visualizations
        print(f"Generating visualizations for run {run_count}...")
        create_visualizations(flats, energies, filename, config, colors_list, sample, emin, fstatemin)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    h = int(elapsed_time / 3600)
    elapsed_time = elapsed_time - (3600 * h)
    m = int(elapsed_time / 60)
    s = elapsed_time - (60 * m)
    
    print(f'\nSimulation completed!')
    print(f'Time elapsed: {h}h {m}m {s:.1f}s')
    print(f'Results saved in: {config.output_dir}/')


def create_visualizations(flats, energies, filename, config, colors_list, sample, emin, fstatemin):
    """
    Create all visualization outputs for a simulation run.
    """
    output_dir = config.output_dir
    
    # Create animation frames
    images = []
    for j in trange(len(flats), desc="Creating animation frames"):
        fig = plt.figure(figsize=(10,10))
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        ax = fig.add_subplot(1, 1, 1)

        # Remove all ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)  # Keep grid lines without numbers
        ax.axis('off')

        coords2 = np.array(flats[j])
        
        # Draw polymer chains
        for i in range(config.p_n):
            ax.plot(coords2[i][:,0],coords2[i][:,1],color='black')
        
        # Draw monomers with different colors
        for i in range(len(config.polymer_sequence)):
            color_idx = i % len(colors_list)
            ax.scatter(coords2[:,i][:,0],coords2[:,i][:,1],s=100,color=colors_list[color_idx])
        
        plt.title('Iteration: %f || Energy: %f'%(((j+1)*sample-1),energies[((j+1)*sample-1)]))
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close()
    
    # Save animation
    output_gif = os.path.join(output_dir, f'{filename}_animation.gif')
    imageio.mimsave(output_gif, images)
    
    # Final state visualization
    fig = plt.figure(figsize=(10,10))
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    ax = fig.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 41, 2)
    minor_ticks = np.arange(0, 41, 1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    coords2 = np.array(flats[-1])
    
    # Draw polymer chains
    for i in range(config.p_n):
        ax.plot(coords2[i][:,0],coords2[i][:,1],color='black')
    
    # Draw monomers with different colors
    for i in range(len(config.polymer_sequence)):
        color_idx = i % len(colors_list)
        ax.scatter(coords2[:,i][:,0],coords2[:,i][:,1],s=100,color=colors_list[color_idx])
    
    energy_info = f"Energy mode: {config.energy_mode}"
    if config.energy_mode == 'same':
        energy_info += f" (value: {config.interaction_energy})"
    
    plt.title(f'{energy_info} || System Energy: {energies[-1]}')
    plt.savefig(os.path.join(output_dir, f'{filename}_final.png'))
    plt.close()
    
    # Energy plot
    fig1 = plt.figure()
    plt.plot(energies)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title(f'Polymer: {config.polymer_sequence} || Energy mode: {config.energy_mode}')
    plt.savefig(os.path.join(output_dir, f'{filename}_energy.png'))
    plt.close()
    
    # Minimum energy state
    if len(fstatemin) > 0:
        fig = plt.figure(figsize=(10,10))
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        ax = fig.add_subplot(1, 1, 1)

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 41, 2)
        minor_ticks = np.arange(0, 41, 1)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax.grid(which='both')

        coords2 = np.array(fstatemin)
        
        # Draw polymer chains
        for i in range(config.p_n):
            ax.plot(coords2[i][:,0],coords2[i][:,1],color='black')
        
        # Draw monomers with different colors
        for i in range(len(config.polymer_sequence)):
            color_idx = i % len(colors_list)
            ax.scatter(coords2[:,i][:,0],coords2[:,i][:,1],s=100,color=colors_list[color_idx])
        
        plt.title(f'Energy mode: {config.energy_mode} || Minimum Energy: {emin}')
        plt.savefig(os.path.join(output_dir, f'{filename}_minim.png'))
        plt.close()


def main():
    """
    Main function to run the simulation.
    """
    parser = argparse.ArgumentParser(description='Polymer Lattice Monte Carlo Simulation')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    config = SimulationConfig()
    
    if args.config:
        config.load_from_file(args.config)
        print(f"Loaded configuration from {args.config}")
    elif args.interactive:
        config.interactive_setup()
    else:
        print("Using default configuration. Use --interactive for custom setup or --config <file> to load settings.")
        config.generate_polymer_sequence()  # Generate default sequence
        print("Default settings:")
        print(f"  Lattice size: {config.l_size}x{config.l_size}")
        print(f"  Polymers: {config.p_n}")
        print(f"  Polymer length: {config.polymer_length}")
        print(f"  Sequence: {config.polymer_sequence}")
        print(f"  Iterations: {config.n_sim}")
        print(f"  Energy mode: {config.energy_mode}")
        print(f"  Energy: {config.interaction_energy}")
    
    # Ask for confirmation
    response = input("\nProceed with simulation? (y/n): ").strip().lower()
    if response != 'y' and response != 'yes':
        print("Simulation cancelled.")
        return
    
    run_simulation(config)


if __name__ == "__main__":
    main()
