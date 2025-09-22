import logging
import pytest
from DFT.DFT_solver import DFTHelium

"""
    Main script for running self-consistent field (SCF) calculations 
    with the radial DFT solver.

    This script initializes the logger, instantiates the solver class,
    and performs the SCF loop for the helium atom (or other atoms if
    parameters are modified). Energies from each iteration are saved
    to a log file, while only the final results are shown on screen.
"""

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#log WARNING message on the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

#log INFO on Helium_DFT.txt
file_handler = logging.FileHandler("Helium_DFT.txt", mode="w")
file_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#initialization of DFTHelium
solver = DFTHelium(Z=2, r_max=30, h=0.001)

# DFT execution
E_new,iters = solver.run()


