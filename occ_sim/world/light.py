"""
light.py
"""

from world.cue import Cue

import numpy as np
import sys

class Light(Cue):
    """
    Class to represent a single light cue (green LED, ersatz sun).
    """
    def __init__(self, elevation=np.pi/4, azimuth=np.pi/2, treatment=None):
        """
        Initialisation.
        :param elevation: The light's elevation
        :param azimuth: The light's azimuth
        :param treatment: The Treatment to which this light cue belongs.
        """
        super().__init__("light", elevation, azimuth, treatment)
        self.__elevation = elevation
        self.__azimuth = azimuth

    #
    # Getters
    #
    def get_elevation(self):
        return self.__elevation

    def get_azimuth(self):
        return self.__azimuth

