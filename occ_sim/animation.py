"""
animation.py

This script is used to procduce animations of population behaviour
over a range of changing conditions. For example, if we wanted to
see how a population would change as light was elevated and wind
kept constant, we could produce the animation and watch the
general trend. This was mostly useful for visualisation, less for
formal analysis.

The script first constructs a series of Treatments which will be
simulated. The simulations are then run, each simulation produces
a plot which is stored in a target directory. These frames can
then be stitched together using a suitable tool, such as ffmpeg.

Note: the script is written for changing light elevations but
it should be reasonably straightforward to modify it for changing
other variables.
"""

from util.deserialiser import Deserialiser

from util.integration_models import *
from util.treatment import Treatment
from util.models import ReliabilityModel

from world.light import Light
from world.wind import Wind

import definitions as defn

import matplotlib.pyplot as plt
from scipy.special import i0
import numpy as np
import os
import shutil

def main():
    #
    # Simulator - can be anything in the util/integration_models module
    #
    simulator = CMLE()

    rel_model = ReliabilityModel()

    #
    # Set the target output directory
    #
    os.chdir("frames/BWS")
    print(os.getcwd())

    start = 30 # Start elevation in degrees
    end = 90 # End elevation

    increment = 1 # Adjustment increment in degrees
    iterations = 15 # Number of simulations to run at each elevation
    treatnent_n = 30 # Number of individuals per treatment

    elevation = np.radians(start)
    filenumber = 0

    wind_speed = 2.5 # Wind speed for each trial (this is assumed to be constant)

    # While elevation still in range
    while elevation < np.radians(end):
        #
        # Create the requisite treatment
        #
        treatment = Treatment()
        treatment.set_reliability_model(rel_model)
        treatment.set_n(treatnent_n)
        treatment.set_id("Elevation {:.01f} degrees".format(np.degrees(elevation)))

        init_light = Light(elevation, np.radians(0), treatment)
        init_wind = Wind(wind_speed, np.radians(0), treatment)
        initial = [init_wind, init_light]

        conf_light = Light(elevation, np.radians(0), treatment)
        conf_wind = Wind(wind_speed, np.radians(120), treatment)
        conflict = [conf_wind, conf_light]

        treatment.set_initial_cues(initial)
        treatment.set_conflict_cues(conflict)

        #
        # Simulate the current treatment for some number of iterations.
        #
        for n in range(iterations):
            #
            # The filename format string is set to produce regular filenames
            # which can easily be stitched into a video using ffmpeg. This can
            # be modified.
            #
            filename = "{:05d}.png".format(filenumber)

            simulator.simulate_treatment(treatment)

            #
            # Plot production
            #
            changes = treatment.get_changes_in_bearing()
            avg_r, avg_t = treatment.get_avg_change()
            plt.tight_layout()
            ax = plt.subplot(121, projection='polar')
            ax.plot(changes, np.ones(len(changes)), 'bo', color='magenta', alpha=0.2)
            ax.plot(avg_t, avg_r, 'ro', markeredgecolor='k', label="R={:.02f},T={:.01f}".format(avg_r, np.degrees(avg_t)))
            ax.set_title(treatment.get_id())
            ax.set_rlim(0,1.1)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.legend(loc='lower left')

            params = treatment.get_cue_distribution_parameters()
            initial_dist_ax = plt.subplot(222)
            initial_light = params["initial"][0]
            initial_wind = params["initial"][1]
            light_mu = initial_light[0]
            wind_mu =  initial_wind[0]
            light_kappa = initial_light[1]
            wind_kappa = initial_wind[1]
            light_x = np.linspace(-np.pi, np.pi, num=100)
            light_y = np.exp(light_kappa*np.cos(light_x - light_mu))/(2*np.pi*i0(light_kappa))
            wind_x = np.linspace(-np.pi, np.pi, num=100)
            wind_y = np.exp(wind_kappa*np.cos(wind_x - wind_mu))/(2*np.pi*i0(wind_kappa))
            initial_dist_ax.plot(np.degrees(light_x), light_y,
                                 color='green',
                                 label="Light: kappa={:.02f}".format(light_kappa)
            )
            initial_dist_ax.plot(np.degrees(wind_x),
                                 wind_y,
                                 color='blue',
                                 label="Wind: kappa={:.02f}".format(wind_kappa))
            initial_dist_ax.set_ylim([0,1])
            initial_dist_ax.legend()
            initial_dist_ax.set_title("Initial cue probability density")
            initial_dist_ax.set_ylabel("Probability density")

            conflict_dist_ax = plt.subplot(224)
            conflict_light = params["conflict"][0]
            conflict_wind = params["conflict"][1]
            light_mu = conflict_light[0]
            wind_mu =  conflict_wind[0]
            light_kappa = conflict_light[1]
            wind_kappa = conflict_wind[1]
            light_x = np.linspace(-np.pi, np.pi, num=100)
            light_y = np.exp(light_kappa*np.cos(light_x - light_mu))/(2*np.pi*i0(light_kappa))
            wind_x = np.linspace(-np.pi, np.pi, num=100)
            wind_y = np.exp(wind_kappa*np.cos(wind_x - wind_mu))/(2*np.pi*i0(wind_kappa))

            conflict_dist_ax.plot(np.degrees(light_x), light_y,
                                  color='green',
                                  label="Light: kappa={:.02f}".format(light_kappa)
            )

            conflict_dist_ax.plot(np.degrees(wind_x),
                                  wind_y, color='blue',
                                  label="Wind: kappa={:.02f}".format(wind_kappa))
            conflict_dist_ax.set_ylim([0,1])
            conflict_dist_ax.set_xlim([-180,180])


            conflict_dist_ax.set_title("Conflict cue probability distributions")
            conflict_dist_ax.set_xlabel("Degrees")
            conflict_dist_ax.set_ylabel("Probability density")

            # Bin data into 360/nbins degree bins to plot the population mass
            nbins = 72
            ch_hist = np.histogram(np.degrees(changes), np.linspace(-180, 180, nbins + 1))[0]
            ch_hist_norm = ch_hist / sum(ch_hist)

            # Plot population response alongside the cue distributions
            plt.bar(np.linspace(-180, 180, nbins),
                    ch_hist_norm, width=360/nbins,
                    color='magenta',edgecolor='k', alpha=0.5,
                    label='Population response')

            conflict_dist_ax.legend()

            plt.gcf().set_size_inches(16,10)
            plt.savefig(filename)
            plt.clf()

            # Loop admin
            filenumber+=1
        elevation+=np.radians(increment)


if __name__ == '__main__':
    main()



