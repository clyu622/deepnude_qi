#!/bin/bash

# Update and Upgrade the Pi, just in case
sudo apt-get update
sudo apt-get upgrade -y

# Install Python3 and Pip if they are not already installed
sudo apt-get install python3
sudo apt-get install python3-pip

# Install necessary Python libraries
pip3 install opencv-python-headless numpy ultralytics flickr_api Pillow opencv-python-headless mediapipe

echo "Installation complete!"
