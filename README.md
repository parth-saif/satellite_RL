# ESA-BIC Project 🛰️🚁🛡️
Welcome to the SAIF Autonomy ESA-BIC project! 

This project consists of 3 parts:

1. Training and evaluating an RL controller for satellite Rendezvous and Proximity Operations (RPO) in simulation. 🛰️
2. Visualising the learned trajectories on a swarm of [Crazyflie 2.1+ drones](https://www.bitcraze.io/products/crazyflie-2-1-plus/ 'Crazyflie drones'). 🚁
3. Implementing and testing a Simplex RTA mechanism for safety assurance for both simulation and drone testing. 🛡️

This work is inspired by the following research [Testing Spacecraft Formation Flying with Crazyflie
Drones as Satellite Surrogates](https://arxiv.org/pdf/2402.14750, 'Barecena et al.'), from the University of Houston.

---
## Setup for Linux or WSL (Ubuntu 22.04)

We use a few key open-source packages in the development of this project.
- [Basilisk](https://avslab.github.io/basilisk/ 'Basilisk') is used as the simulation engine for satellite dynamics.
- [BSK-RL](https://avslab.github.io/bsk_rl/index.html 'bsk-rl') is a package for constructing Gymnasium environments for spacecraft tasking problems, using Basilisk.
- [Crazyflie Python API](https://github.com/bitcraze/crazyflie-lib-python 'Crazyflie Python API') is used to send instructions to the Crazyflie 2.1+ drones.

All the required packages can be installed with the following instructions:

1. Clone this repository to the desired folder.
2. Create a Python virtual environment:
   - **Python 3.10**
   - Use **pip + venv**. DO NOT use conda, Basilisk will not build.
3. In you folder, and while your virtual environment is active, install the required pacakges in the terminal:
   ```
   ./setup_packages.sh
   ```
   Now you have everything you need to run simulations and send instructrions to the swarm!
4. To setup the Crazyflie 2.1+ drones, and radio communication, refer to further documentation and [online resources](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/ 'Get started with the Crazyflie 2.1+')
5. To run existing examples, refer to the 'tests/' folder.

---
## Repositry outline

### Main code
- `src/` - main python programs used to setup and run simulations, as well automated drone flying scripts.
- `tests/` - python programs to test specific parts and quickstart testing.
- `models/` - .pth Neural Network models for RL.
- `notebooks/` - Juypter notebooks for analysis.

### Visualisation
- `viz_output/` - Vizard binary files for post-visualisation satellite simulations.
- `images/` - nice images for reference.

### Utilities
- `config/` - .yaml files to configure various simulations and drone tests.
- `data/` - useful data like trajectories, logs etc.
- `scripts/` - installation script.

---
## Visualisation

To visualise the satellite simulations, you need to install **Vizard**. The installation instructions can be found [here](https://hanspeterschaub.info/basilisk/Vizard/VizardDownload.html 'Vizard installation'). This allows Unity-based 3D visualisation.

![Satellite view](./images/satellite.png 'Satellite')

![Earth view](./images/earth_sats.png 'Earth view')

## Documentation
Further documentation is to be added to Notion.
