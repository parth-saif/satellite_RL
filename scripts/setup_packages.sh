#!/bin/bash

# This script automates the process of downloading and installing necessary packages for the ESA-BIC project from source.
# It is designed to be run in a Linux environment - specifically Ubuntu 22.04 is proven to work.

# --- Configuration ---
# Define the project root relative to where this script is executed.
# Assuming this script is in 'scripts/' and you run it from the project root.
PROJECT_ROOT=$(pwd)

# Define the directory where external packages will be stored.
# This should be the 'packages/' directory in your project root.
EXTERNAL_PACKAGES_DIR="$PROJECT_ROOT/packages"

# --- Setup ---
echo "--- Starting external package setup ---"

# Create necessary directories if they don't exist
mkdir -p "$EXTERNAL_PACKAGES_DIR"

# install neceassary dependencies
echo "--- Installing necessary dependencies ---"
sudo apt-get update
sudo apt-get install -y \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-tk \
    swig

# --- Basislik ---
echo "--- Installing Basilisk ---"
PACKAGE1_NAME="basilisk"
PACKAGE1_REPO="https://github.com/AVSLab/basilisk.git"
PACKAGE1_SRC_DIR="$EXTERNAL_PACKAGES_DIR/$PACKAGE1_NAME"

# Build Basilisk
echo "Building $PACKAGE1_NAME..."
# First intstall python dependencies using pip
pip install --upgrade pip 
pip install wheel 'conan>2.0' cmake

# Check if source directory already exists
if [ -d "$PACKAGE1_SRC_DIR" ]; then
    echo "Source for $PACKAGE1_NAME already exists. Pulling latest changes..."
else
    echo "Cloning $PACKAGE1_NAME source from $PACKAGE1_REPO..."
    git clone "$PACKAGE1_REPO" "$PACKAGE1_SRC_DIR" || { echo "Error: Failed to clone $PACKAGE1_NAME."; exit 1; }
    # Then build Basilisk
    cd "$PACKAGE1_SRC_DIR"
    python3 conanfile.py --clean
fi

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to build $PACKAGE1_NAME."
    exit 1
else
    echo "$PACKAGE1_NAME installed successfully."
fi

# --- BSK-RL ---
echo "--- Installing BSK-RL ---"
PACKAGE2_NAME="bsk_rl"
PACKAGE2_REPO="https://github.com/AVSLab/bsk_rl.git"
PACKAGE2_SRC_DIR="$EXTERNAL_PACKAGES_DIR/$PACKAGE2_NAME"
# Check if source directory already exists
if [ -d "$PACKAGE2_SRC_DIR" ]; then
    echo "Source for $PACKAGE2_NAME already exists. Pulling latest changes..."
else
    echo "Cloning $PACKAGE2_NAME source from $PACKAGE2_REPO..."
    git clone "$PACKAGE2_REPO" "$PACKAGE2_SRC_DIR" || { echo "Error: Failed to clone $PACKAGE2_NAME."; exit 1; }
    cd "$PACKAGE2_SRC_DIR"
    python -m pip install -e "." ".[rllib]" && finish_install
fi  

# Check if the installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to install $PACKAGE2_NAME."
    exit 1
else
    echo "$PACKAGE2_NAME installed successfully."
fi

# --- Crazyflie ---
echo "--- Installing Crazyflie Python API---"
pip install cfclient # this also installs cflib 
# Check if the installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Crazyflie Python API."
    exit 1
else
    echo "Crazyflie Python API installed successfully."
fi

# --- Trajectory Generation Package ---
echo "--- Installing Trajectory Generation Package (uav_trajectories) ---"
PACKAGE3_NAME="uav_trajectories"
PACKAGE3_REPO="https://github.com/whoenig/uav_trajectories.git"
PACKAGE3_SRC_DIR="$EXTERNAL_PACKAGES_DIR/$PACKAGE3_NAME"

# Install dependencies for this package
sudo apt-get install -y libeigen3-dev libboost-program-options-dev libboost-filesystem-dev libnlopt-cxx-dev libgoogle-glog-dev

# Check if source directory already exists
if [ -d "$PACKAGE3_SRC_DIR" ]; then
    echo "Source for $PACKAGE3_NAME already exists. Skipping."
else
    echo "Cloning $PACKAGE3_NAME source from $PACKAGE3_REPO..."
    git clone "$PACKAGE3_REPO" "$PACKAGE3_SRC_DIR" || { echo "Error: Failed to clone $PACKAGE3_NAME."; exit 1; }
    # Build the package
    echo "Building $PACKAGE3_NAME..."
    cd "$PACKAGE3_SRC_DIR"
    mkdir -p build
    cd build
    cmake ..
    make
fi

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to build $PACKAGE3_NAME."
    exit 1
else
    echo "$PACKAGE3_NAME built successfully."
fi
cd "$PROJECT_ROOT" # Return to project root

echo "--- All packages installed successfully ---"
