"""
models.py

This module contains the tools to perform r-value approximation from
the governing cue property (e.g. elevation or wind-speed).

In code, the method full method for kappa approximation is fragmented.
R-values are computed here, r-augmentation and kappa approximation happen
in world/cue.py.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import optimize
from scipy import stats

import pandas as pd
import sys

#
# Precision data
#
wind_rvals = pd.read_csv("data/wind_rvals.csv")
elevation_rvals = pd.read_csv("data/elevation_rvals.csv")

# Remove taxis conditions
wind_rvals = wind_rvals[wind_rvals.columns[3:7]]
wind_speeds = [float(x.split(" ")[0]) for x in wind_rvals.columns]
wind_rvals.columns = wind_speeds # Rename columns to omit m/s

elevations = [int(x.split("d")[0]) for x in elevation_rvals.columns]
elevation_rvals.columns = np.radians(elevations)

# Extract means for each cue configuration
wind_meds = wind_rvals.mean()
elevation_meds = list(elevation_rvals.mean())

# Median R values for upper (>75) and lower elevations
split=5
light_lower = np.radians(elevations[:split]) # Elevations from 5 to 75 inclusive
light_upper = np.radians(elevations[split-1:]) # Elevations from 75 to 90 inclusive
light_lower_meds = elevation_meds[:split]
light_upper_meds = elevation_meds[split-1:]

class ReliabilityModel():
    """
    ReliabilityModel contains the model(s) required to convert cue configurations
    to reliabilities.
    """
    def __init__(self):
        """
        Constructor; define models of reliability for wind, lower light
        elevations, and upper light elevations.
        """
        self.wind_params, self.params_covariance = optimize.curve_fit(self.linear, wind_speeds, wind_meds)
        self.lower_light_params, _ = optimize.curve_fit(self.linear,
                                                        light_lower,
                                                        light_lower_meds)
        self.upper_light_params, _ = optimize.curve_fit(self.linear,
                                                        light_upper,
                                                        light_upper_meds)

        # Print the linear parameters.
        print("Parameters")
        print("Wd: {}".format(self.wind_params))
        print("LL: {}".format(self.lower_light_params))
        print("UL: {}".format(self.upper_light_params))

    def quartic(self, x, a, b, c, d, e):
        """
        [Legacy] Quartic curve for use with curve_fit()
        """
        return a * x + b * x ** 2 + c * x**3 + d * x ** 4 + e

    def cubic(self, x, a, b, c, d):
        """
        [Legacy] Cubic curve for use with curve_fit()
        """
        return a * x + b * x ** 2 + c * x**3 + d

    def proj_curve(self, x, scale, xscale, vshift):
        """
        [Legacy] curve_fit compatible function based on the geometric
        relationship between light elevation and theoretically available
        directional information.

        Draw a vector from the agent to the light source, then project this
        into the ground plane. scale, hshift, and vshift are included to allow
        the curve_fit function to try a variety of configurations.
        """
        return scale * np.cos(xscale*x) + vshift

    def linear(self, x, a, b):
        """
        Linear function for use with curve_fit()
        """
        return a*x + b


    def convert_general(self, cue_type, independent_var, params=None):
        """
        Given a cue-type and governing variable, use the appropriate linear model
        to return a raw R value.

        :param cue_type: String, cue type; expects light or wind
        :param independent_var: the value of either the wind speed, or
                                light elevation (in radians)
        """
        if cue_type == "wind":
            return self.convert_wind_speed(independent_var)
        elif cue_type == "light":
            return self.convert_light_elevation(independent_var)

        return 0 # catch-all

    def convert_wind_speed(self, wind_speed):
        """
        Convert wind speed to a raw R value.
        :param wind_speed: wind speed in m/s
        """
        a, b = self.wind_params
        return self.linear(wind_speed, a, b)

    def convert_light_elevation(self, light_elevation):
        """
        Convert light elevatino to a raw R value.
        :param light_elevation: Angular elevation in radians
        """
        a = 0
        b = 0

        if light_elevation < np.radians(76):
            a, b = self.lower_light_params
        else:
            a, b = self.upper_light_params

        return self.linear(light_elevation, a, b)

    def get_wind_params(self):
        """
        Getter, for the linear wind parameters
        """
        return self.wind_params

    def get_lower_light_params(self):
        """
        Getter for the linear light elevation (<=75deg) parameters
        """
        return self.lower_light_params

    def get_upper_light_params(self):
        """
        getter for the linear light elevation (>75deg) parameters
        """
        return self.upper_light_params

reliability_model = ReliabilityModel()

if __name__ == "__main__":
    """
    Used for testing and plotting to find reasonable fits for the data; the final
    iteration was used to produce Figure 1 from the paper.
    """
    rel_model = ReliabilityModel()

    # Size figure for A4
    fig = plt.figure(figsize=(10,3))
    plt.tight_layout()
    plt.rc('axes',labelsize=12)
    plt.rc('axes',titlesize=12)
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    plt.rc('legend',fontsize=8)
    plt.suptitle("Orientation precision with increasing light elevation and wind speed",y=0.96)

    #
    # Wind subplot
    #
    ax = plt.subplot(1, 3, 3)

    # Linear estimator datapoints
    windvals = np.linspace(1.0,2.5,100)
    wind_ys = np.array(rel_model.convert_wind_speed(windvals))

    # Plot means with std error instead of boxplots
    wind_means = np.mean(wind_rvals) # Means
    wind_sem = stats.sem(wind_rvals) # Standard error

    # Wind estimator
    ax.errorbar(wind_speeds,
                wind_means,
                yerr=wind_sem,
                fmt='.',
                label="Mean precision"
    )

    ax.plot(windvals,
            wind_ys,
            color='tab:blue',
            label="Linear precision estimator",
            alpha=0.3,
            zorder=4)

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylim([0, 1])
    ax.set_yticks([])
    ax.set_xticks(np.arange(1,3,0.5))
    ax.set_xlim([0.5, 3.0])

    #
    # Elevation subplot
    #
    ax2 = plt.subplot(1, 3, (1,2))

    # Summary statistics from data
    elevation_means = np.mean(elevation_rvals)
    elevation_sem = stats.sem(elevation_rvals)

    # x data points for lower and upper elevation estimator plots
    lower_elevations = np.linspace(5, 76, 1000, endpoint=False)
    upper_elevations = np.linspace(76, 90, 1000)

    # y data points for lower and upper elevation estimator plots
    lower_rvals = [rel_model.convert_light_elevation(x) for x in np.radians(lower_elevations)]
    upper_rvals = [rel_model.convert_light_elevation(x) for x in np.radians(upper_elevations)]

    # Omit 82, 84, 86, and 88 for readability at A4 scale
    ticklbls = ['5', '20', '45', '60', '75', '80','','', '', '','90']
    ax2.set_xticklabels(ticklbls)

    # Plot data summary stats
    ax2.errorbar(elevations,
                 elevation_means,
                 yerr=elevation_sem,
                 fmt='.',
                 color='forestgreen',
                 label="Mean precision")

    # Plot lower and upper estimators
    est_colour = 'tab:green'
    est_alpha = 0.3
    ax2.plot(lower_elevations,
             lower_rvals,
             color=est_colour,
             alpha=est_alpha,
             label='Split-linear precision estimator')
    ax2.plot(upper_elevations,
             upper_rvals,
             color=est_colour,
             alpha=est_alpha)

    # Highlighting 60 and 75 degrees - replot and add horizontal lines
    lw = 1
    sx = np.mean(elevation_rvals[np.radians(60)])
    sf = np.mean(elevation_rvals[np.radians(75)])

    ax2.errorbar(60,
                 sx,
                 yerr=elevation_sem[3],
                 fmt='.',

                 color='darkred')
    ax2.errorbar(75,
                 sf,
                 yerr=elevation_sem[4],
                 fmt='.',
                 linestyle='-.',
                 color='indianred')

    ax2.axhline(y=sx,
                color='darkred',
                xmin=60/92,
                linestyle='--',
                alpha=1,
                zorder=0,
                linewidth=lw)

    ax2.axhline(y=sf,
                color='indianred',
                linestyle='-.',
                xmin=75/92,
                alpha=0.5,
                zorder=0,
                linewidth=lw)

    ax2.set_xlabel("Light elevation (degrees)")
    ax2.set_ylabel("R value")
    ax2.set_ylim([0,1])
    ax2.set_xlim([4,91])
    ax2.set_xticks(elevations)
    ax2.legend()

    #
    # Add horizontal lines to the wind axis
    #

    # Plot means at 60, 75
    sx = np.mean(elevation_rvals[np.radians(60)])
    sf = np.mean(elevation_rvals[np.radians(75)])

    ax.axhline(y=sx,
               color='darkred',
               linestyle='--',
               alpha=1,
               zorder=0,
               linewidth=lw)

    ax.axhline(y=sf,
               color='indianred',
               linestyle='-.',
               alpha=0.5,
               zorder=0,
               linewidth=lw)


    ax.legend()
    fig.subplots_adjust(wspace=0.03)

    """
    Uncomment to display or reproduce figure.
    """
#    plt.savefig("kappa_estimators.pdf", bbox_inches="tight")
#    plt.show()
