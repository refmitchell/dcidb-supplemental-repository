"""
cue.py

Much of the underlying functionality of a "cue" is shared. A superclass
is provided here to reduce code duplication. This does break down in
places (where the type needs to be checked).
"""
from scipy.special import iv
from scipy.optimize import root_scalar
import numpy as np
import sys

class Cue():
    """
    Class to store basic shared properties of cues.
    """

    def __kappa_mle(self, k, R):
        """
        Maximum likelihood estimate of von-Mises kappa parameter
        from (Mardia and Jupp, 2000, pg. 85 (5.3.5); approximations
        available on pg. 85/86. The root of this function is the MLE
        of kappa (use scipy optimize). May be used as a point of
        comparison; slightly slower for general use.

        :param k: kappa value
        :param R: R value
        :return: result of A(k) - R (See Mardia and Jupp).
        """
        return (iv(1, k) / iv(0, k)) - R

    def __kappa_approximation(self, R):
        """
        Kappa MLE approximation from (Mardia and Jupp, 2000, pg. 85,86).

        :param R: the mean vector length R
        :return: an approximation of the MLE of kappa
        """
        #
        # R augmentation stage; these are the constants c_Wind and c_Light
        #
        light_adj = 0.135
        wind_adj = 0.133
        if self.__type == "wind":
            R += wind_adj
        if self.__type == "light":
            R += light_adj

        #
        # Kappa approximation, cases
        #

        # For "small" R (5.3.7); R < 0.53
        if R < 0.53:
            return 2*R + R**3 + (5/6)*(R**5)

        # For "large" R (5.3.8); R >= 0.85
        if R >= 0.85:
            return 1 / (2*(1-R) - ((1 - R)**2) - ((1 - R)**3))

        # For "medium" R (5.3.10); 0.53 <= R < 0.85
        return -0.4 + 1.39*R + (0.43/(1-R))

    def __init__(self, cue_type, independent_variable, mean, treatment):
        """
        Constructor
        :param cue_type: String, cue type; expects "wind" or "light"
        :param independent_variable: Elevation (rad) for light, wind speed for
                                     wind.
        :param mean: The mean of the von Mises distribution (cue azimuth)
        :param treatment: The Treatment to which this cue belongs
        """
        if treatment == None:
            sys.exit("Fatal: cue without a treatment.")
        elif treatment.get_reliability_model() == None:
            sys.exit("Fatal: reliability model is NoneType. Hint: You're"
                     " probably trying to construct cues before the model"
                     " has been added to the treatment. Order matters.")

        self.__type = cue_type
        self.__model = treatment.get_reliability_model()
        self.__id_var = independent_variable

        self.__mu = mean # for von-Mises

        # Mean vector length
        self.__R = self.__model.convert_general(self.__type,
                                                self.__id_var
                                                )

        # Kappa approximation
        self.__kappa = self.__kappa_approximation(self.__R)
        self.__treatment = treatment

    #
    # Getters
    #
    def get_type(self):
        return self.__type

    def get_reliability(self):
        return self.__R

    def get_weight(self):
        return self.__kappa

    def get_azimuth(self):
        return self.__mu

    def get_vm_parameters(self):
        """
        Get von Mises parameters
        """
        return (self.__mu, self.__kappa)

    def sample(self, n=1):
        """
        Draw n random samples from the von Mises distribution characterised by
        this cue.
        """
        return np.random.vonmises(self.__mu, self.__kappa, n) if n > 1 else np.random.vonmises(self.__mu, self.__kappa)



