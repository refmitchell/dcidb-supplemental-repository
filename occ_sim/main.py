"""
main.py

Used as the main development script for the simulator before
evaluation and generally used for experimentation with full simulation
setups; as such this script is not particularly cohesive. This may be
useful for working with custom configurations of simulator or
treatment.

If you want a quick start with the simulator it is recommended to use
occ_sim.py.
"""

# Internal imports
from util.deserialiser import Deserialiser
from util.integration_models import CMLE
from util.integration_models import WTA
from util.integration_models import SingleCue
from util.integration_models import BWS
from util.parallax_model import GeometricSimulator

from util.treatment import Treatment
from util.models import ReliabilityModel
from world.light import Light
from world.wind import Wind

import definitions as defn

# Python library imports
import matplotlib.pyplot as plt
from scipy.special import i0
import numpy as np
import os
import shutil
import pandas as pd

def main(config_file=""):
    # If a config file is specified; config files were initially used
    # but became more of a hindrance as development continued. They should
    # still be usable for basic simulations.
    if config_file != "":
        defn.CONFIG_FILE = os.path.join(defn.CONFIG_DIR, config_file)

    deserialiser = Deserialiser(configuration_path=defn.CONFIG_FILE)

    treatments = deserialiser.init_configuration()

    ########################################################
    # Simulation setup - no config file                    #
    ########################################################
    elevation = 86
    wind_speed = 2.5
    conflict = 120

    rel_model = ReliabilityModel()
    treatment = Treatment()
    treatment.set_reliability_model(rel_model)
    treatment.set_n(30)
    treatment.set_id("Weightings")

    init_light = Light(np.radians(elevation), np.radians(0), treatment)
    init_wind = Wind(wind_speed, np.radians(0), treatment)
    initial = [init_wind, init_light]

    conf_light = Light(np.radians(elevation), np.radians(0), treatment)
    conf_wind = Wind(wind_speed, np.radians(conflict), treatment)
    conf= [conf_wind, conf_light]

    treatment.set_initial_cues(initial)
    treatment.set_conflict_cues(conf)
#    treatment.generate_samples()
    ########################################################

    #
    # Set simulator
    #

#    simulator = SingleCue(cue="wind")
#    simulator = WTA()

#    simulator = BWS(bias_window=-1, adjustment_slope=41, adjustment_bias=0.36)
#    simulator = GeometricSimulator()
#    simulator = CMLE()

    sigma = 0
    iters = 1
    for i in range(iters):
        paths = simulator.simulate_treatment(treatment)

        # init_paths = paths[0]
        # conf_paths = paths[1]

        # rs = [ p[0] for p in init_paths[0] ]
        # ts = [ p[1] for p in init_paths[0] ]

        # crs = [ p[0] for p in conf_paths[0] ]
        # cts = [ p[1] for p in conf_paths[0] ]

        changes = treatment.get_changes_in_bearing()

        deg_changes = np.degrees(changes)
        rounded_changes = [ int(5 * round(x/5)) for x in deg_changes ]

        avg_r, avg_t = treatment.get_avg_change()

        sigma += avg_r
        print("R: " + str(avg_r))
        print("Mu: " + str(np.degrees(avg_t)))

        # os.chdir("mimic_distributions")
        # df = pd.DataFrame()
        # df["changes"] = changes
        # df["deg_changes"] = deg_changes
        # df["rounded"] = rounded_changes
        # df["modulo"] = [x % 360 for x in rounded_changes ]
        # df.to_csv(
        #     "{}.csv".format(treatment.get_id())
        # )


        # Bin data into 360/nbins degree bins to plot the population mass
        nbins = 72
        ch_hist = np.histogram(np.degrees(changes), np.linspace(-180, 180, nbins + 1))[0]
        ch_hist_norm = ch_hist / sum(ch_hist)

        ch_hist_shift = np.histogram(np.degrees(changes) % 360, np.linspace(0,360, nbins + 1))[0]
        # print(ch_hist)
        # print(ch_hist_shift)
        # print(np.degrees(changes))
        # print(np.degrees(changes) % 360)

        ax = plt.subplot(121, projection='polar')
        ax.plot(changes, np.ones(len(changes)), 'bo', color='magenta', alpha=0.2)
        ax.plot(avg_t, avg_r, 'ro', markeredgecolor='k', label="R={:.02f},Th={:.01f}".format(avg_r, np.degrees(avg_t)))

#        ax.plot(ts, rs)
#        ax.plot(cts, crs)

        ax.set_title("{}: {}".format(simulator.name(), treatment.get_id()))
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
        initial_dist_ax.plot(np.degrees(light_x), light_y, color='green', label="Light: kappa=" + str(light_kappa))
        initial_dist_ax.plot(np.degrees(wind_x), wind_y, color='blue', label="Wind: kappa=" + str(wind_kappa))
        initial_dist_ax.set_ylim([0,1])
        initial_dist_ax.legend()
        initial_dist_ax.set_title("Initial cue probability distributions")
        initial_dist_ax.set_xlabel("Degrees")
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

        # Plot population response
        plt.bar(np.linspace(-180, 180, nbins), ch_hist_norm, width=360/nbins, color='magenta',edgecolor='k', label='Population response', alpha=0.5)

        conflict_dist_ax.plot(np.degrees(light_x), light_y, color='green', label="Light: kappa=" + str(light_kappa))
        conflict_dist_ax.plot(np.degrees(wind_x), wind_y, color='blue', label="Wind: kappa=" + str(wind_kappa))
        conflict_dist_ax.set_ylim([0,1])
        conflict_dist_ax.set_xlim([-180,180])
        conflict_dist_ax.legend()

        conflict_dist_ax.set_title("Conflict cue probability distributions")
        conflict_dist_ax.set_xlabel("Degrees")
        conflict_dist_ax.set_ylabel("Probability density")
        plt.show()

    avg = sigma/iters
    print("Average:{}".format(avg))

if __name__ == '__main__':
    main("config.yaml")



