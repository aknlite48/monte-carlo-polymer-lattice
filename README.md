# Polymer Lattice Monte Carlo Simulation

A Python implementation of Monte Carlo simulation for polymer chains on a 2D lattice, featuring interactive visualization and energy analysis.

## Overview

This simulation models polymer chains as sequences of connected monomers on a discrete 2D lattice. Using Monte Carlo methods, it explores different conformations to find energetically favorable states. The simulation includes real-time visualization, energy tracking, and statistical analysis.

## Features

- **Monte Carlo Simulation**: Efficient sampling of polymer conformations using Metropolis algorithm
- **Interactive Setup**: Command-line interface for easy parameter configuration
- **Real-time Visualization**: Animated GIF generation showing polymer evolution
- **Energy Analysis**: Comprehensive energy tracking and minimum energy state detection
- **Flexible Configuration**: JSON-based configuration files for reproducible simulations
- **Multiple Energy Modes**: Support for uniform or monomer-specific interaction energies
- **Batch Processing**: Multiple simulation runs with automated output organization

## Installation

### Prerequisites

```bash
pip install numpy matplotlib celluloid tqdm imageio
```

### Required Libraries

- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `celluloid`: Animation creation
- `tqdm`: Progress bars
- `imageio`: GIF generation

## Quick Start

### Basic Usage

Run with default parameters:
```bash
python polymer_simulation.py
```

### Interactive Mode

Set up custom parameters interactively:
```bash
python polymer_simulation.py --interactive
```

### Configuration File

Use a predefined configuration:
```bash
python polymer_simulation.py --config my_config.json
```

## Configuration

### Default Parameters

- **Lattice Size**: 20×20 grid
- **Polymer Chains**: 6 chains
- **Polymer Length**: 4 monomers (sequence: 'abcd')
- **Iterations**: 1000 Monte Carlo steps
- **Energy Mode**: Same energy for all interactions (-1.0)
- **Temperature**: kT = 1.0
- **Sampling Rate**: Save every 10th frame

### Configuration File Format

```json
{
  "l_size": 20,
  "p_n": 6,
  "polymer_length": 4,
  "n_sim": 1000,
  "interaction_energy": -1.0,
  "energy_mode": "same",
  "custom_energies": {},
  "sampling_rate": 10,
  "n_runs": 1,
  "kt": 1.0,
  "output_dir": "simulation_output"
}
```

### Energy Modes

1. **Same Energy Mode**: All monomer interactions have identical energy
2. **Custom Energy Mode**: Individual energy values for each monomer type

## Output Files

Each simulation run generates:

- `*_animation.gif`: Animated visualization of polymer evolution
- `*_final.png`: Final polymer configuration
- `*_energy.png`: Energy vs. iteration plot
- `*_minim.png`: Minimum energy configuration found
- `config.json`: Complete simulation parameters

## Algorithm Details

### Monte Carlo Procedure

1. **Random Selection**: Pick a random monomer from any polymer chain
2. **Move Generation**: Find all valid moves maintaining chain connectivity
3. **Energy Calculation**: Compute energy change for the proposed move
4. **Acceptance Criterion**: Accept/reject based on Metropolis criterion:
   - Always accept if ΔE ≤ 0
   - Accept with probability exp(-ΔE/kT) if ΔE > 0

### Constraints

- **Chain Connectivity**: Monomers must remain connected to neighbors
- **Excluded Volume**: No two monomers can occupy the same lattice site
- **Boundary Conditions**: Polymers are confined within the lattice boundaries

## Classes and Methods

### `PolymerLatticeSimulation`

Main simulation class handling lattice operations:

- `embed(p_n, poly)`: Randomly place polymer chains on lattice
- `latenergy(energy)`: Calculate total system energy
- `avmoves(pos)`: Find valid moves for a monomer
- `move(ipos, fpos)`: Execute monomer movement
- `visual()`: Generate visualization data

### `SimulationConfig`

Configuration management class:

- `interactive_setup()`: Interactive parameter input
- `load_from_file()`: Load configuration from JSON
- `save_to_file()`: Save configuration to JSON
- `setup_energy_matrix()`: Generate interaction energy matrix

## Example Usage

### Basic Simulation

```python
from polymer_simulation import SimulationConfig, run_simulation

# Create configuration
config = SimulationConfig()
config.l_size = 15
config.p_n = 4
config.polymer_length = 6
config.n_sim = 2000

# Run simulation
run_simulation(config)
```

### Custom Energy Configuration

```python
config = SimulationConfig()
config.energy_mode = 'custom'
config.custom_energies = {
    'a': -2.0,
    'b': -1.5,
    'c': -1.0,
    'd': -0.5
}
run_simulation(config)
```

## Visualization

The simulation generates several types of visualizations:

- **Animation**: Shows polymer movement over time with different colors for each monomer type
- **Energy Plot**: Tracks energy evolution throughout the simulation
- **Final State**: Static image of the final polymer configuration
- **Minimum Energy State**: Configuration with the lowest energy encountered

## Performance Considerations

- **Lattice Size**: Larger lattices require more memory and computation time
- **Polymer Length**: Longer polymers have fewer valid moves, potentially slowing convergence
- **Sampling Rate**: Higher sampling rates create smoother animations but larger files
- **Number of Iterations**: More iterations provide better sampling but increase runtime

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required packages are installed
2. **Memory Issues**: Reduce lattice size or sampling rate for large simulations
3. **Slow Convergence**: Increase temperature (kT) or reduce interaction energies
4. **Animation Problems**: Check that imageio is properly installed

### Performance Tips

- Use smaller lattices for initial testing
- Reduce sampling rate for faster execution
- Monitor energy plots to assess convergence
- Use multiple shorter runs instead of one long run

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.

## License

This project is open source. Please cite appropriately if used in academic work.

## References

- Monte Carlo Methods in Statistical Physics
- Polymer Physics and Computational Modeling
- Lattice Models in Polymer Science
