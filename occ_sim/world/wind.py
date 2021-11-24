"""
wind.py
"""

from world.cue import Cue

import numpy as np
import sys

class Wind(Cue):
    """
    Class to represent a wind cue.
    """
    def __init__(self, speed=2.5, azimuth=np.pi/2, treatment=None):
        """
        Wind cue initialisation
        :param speed: The speed of the wind in m/s
        :param azimuth: The prevailing angular direction of the wind.
        :param treatment: The Treatment to which this wind cue belongs
        """
        super().__init__("wind", speed, azimuth, treatment)

