"""
data_production.py

This script was used to produce the mimic data from the paper; as
noted in the text, this data was specifically curated to be as close
to the biological data as possible using mean vector error. As stated
in the text data is meant to demonstrate that the observed data is
possible under the chosen model, not how likely it is to occur.
"""

from util.treatment import Treatment
from util.models import ReliabilityModel
from util.integration_models import *

import pandas as pd

from world.wind import Wind
from world.light import Light

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def circ_mean(angles):
    """
    Compute the circular mean of a list of angles. Standard method
    is to compute angles to unit vectors, then take the vector average.
    See Batschelet (1981) for more information.

    :param angles: List of angles to average.
    :return: The circular mean.
    """
    xs = np.cos(angles)
    ys = np.sin(angles)
    avg_x = sum(xs) / len(xs)
    avg_y = sum(ys) / len(ys)
    avg_r = np.sqrt(avg_x**2 + avg_y**2)
    avg_t = np.arctan2(avg_y, avg_x)
    return (avg_r, avg_t)

def polar_euclidean_distance(a,b):
    """
    Euclidean distance between two polar vectors (in tuple form)
    :param a: Tuple of form (r, theta)
    :param b: Tuple of form (r, theta)
    :return: Euc distance between a and b
    """
    a_cart = (a[0] * np.cos(a[1]),
              a[0] * np.sin(a[1]))
    b_cart = (b[0] * np.cos(b[1]),
              b[0] * np.sin(b[1]))

    return np.sqrt(((a_cart[0] - b_cart[0]) ** 2) +  ((a_cart[1] - b_cart[1]) ** 2))


def cue_weight_conditions(sim):
    """
    Generate mimic data for the cue weighting experiments. 1.25 and
    2.5m/s wind speed across all elevations and conflicts.

    :param sim: The simulator with which to produce the data.
    :return: Unused
    """

    # Data input; this is the biological data used for approximation.
    data = pd.read_csv("data/cue_weight_data.csv")

    # Generate all experimental conditions
    elevations = [45, 60, 75, 86]
    conflicts = [0, 60, 120]
    windspeeds = [1.25, 2.5]
    treatments = dict()

    rel_model = ReliabilityModel()

    simulator = sim

    #
    # Treatment production
    #
    for wind_speed in windspeeds:
        for elevation in elevations:
            # Condition doesn't exist so don't generate unnecessary work.
            if elevation == 45 and wind_speed == 1.25:
                print("Skipping condition {}-{}".format(
                    wind_speed, elevation
                ))
                continue
            for conflict in conflicts:
                key = "{:.02f}-{}-{:03d}".format(wind_speed, elevation, conflict)
                n = len([x for x in list(data[key]) if not np.isnan(x)])

                print("{}:{}".format(key,n))

                # Generate treatment for each condition with extremely
                # large sample sizes to get a better picture of the
                # parent population.
                treatment = Treatment()
                treatment.set_reliability_model(rel_model)
                treatment.set_n(n)
                treatment.set_id("E:{};C:{}".format(elevation, conflict))

                init_light = Light(np.radians(elevation), np.radians(0), treatment)
                init_wind = Wind(wind_speed, np.radians(0), treatment)
                initial = [init_wind, init_light]

                conf_light = Light(np.radians(elevation), np.radians(0), treatment)
                conf_wind = Wind(wind_speed, np.radians(conflict), treatment)
                conf= [conf_wind, conf_light]

                treatment.set_initial_cues(initial)
                treatment.set_conflict_cues(conf)

                treatments[key] = treatment

    print("Treatment generation complete")

    # Move into mimic directory for output storage
    os.chdir("mimic_distributions/cue_weight")
    print(os.getcwd())

    #
    # For each treatment, search for a sufficiently similar simulated population
    #
    for key in treatments.keys():
        # Compute the target mean
        print("Working on {}".format(key))
        condition_data = [x for x in list(data[key]) if not np.isnan(x)]
        cond_avg = circ_mean(np.radians(condition_data))

        # Simulated summary stats
        treatment = treatments[key]

        #
        # Search for sufficiently low error, break when found.
        #
        while True:
            simulator.simulate_treatment(treatment)
            changes = treatment.get_changes_in_bearing()
            deg_changes = np.degrees(changes)
            rounded_changes = [ int(5 * round(x/5)) for x in deg_changes ]
            sim_avg = circ_mean(np.radians(rounded_changes))

            error = polar_euclidean_distance(cond_avg, sim_avg)

            # Chosen error threshold, chosen to be sufficiently similar
            if error < 0.02:
                break

        #
        # Store the simulated population information
        #
        df = pd.DataFrame()
        df["changes"] = changes
        df["deg_changes"] = deg_changes
        df["rounded"] = rounded_changes
        df["modulo"] = [x % 360 for x in rounded_changes ]
        df.to_csv(
            "{}.csv".format(key)
        )

    return False


def sixty_multi_day_conditions(sim):
    """
    Generate mimic data for the sixty degree conditions over multiple days.

    :param sim: The simulator in use.
    """
    data = pd.read_csv("data/data_sixty.csv")

    # Generate all experimental conditions
    days = [0, 1, 2]
    conflicts = [0, 60, 120]
    treatments = dict()

    elevation=60
    wind_speed=2.5

    rel_model = ReliabilityModel()

    simulator = sim

    #
    # Treatment generation
    #
    for day in days:
        for conflict in conflicts:
            key = "{}-2.50-60-{:03d}".format(day, conflict)
            n = len([x for x in list(data[key]) if not np.isnan(x)])
            print("{}:{}".format(key,n))

            treatment = Treatment()
            treatment.set_reliability_model(rel_model)
            treatment.set_n(n)
            treatment.set_id("E:{};C:{}".format(elevation, conflict))

            init_light = Light(np.radians(elevation), np.radians(0), treatment)
            init_wind = Wind(wind_speed, np.radians(0), treatment)
            initial = [init_wind, init_light]

            conf_light = Light(np.radians(elevation), np.radians(0), treatment)
            conf_wind = Wind(wind_speed, np.radians(conflict), treatment)
            conf= [conf_wind, conf_light]

            treatment.set_initial_cues(initial)
            treatment.set_conflict_cues(conf)

            treatments[key] = treatment

    print("Treatment generation complete")

    # Move into mimic directory; for output storage.
    os.chdir("mimic_distributions/sixty")
    print(os.getcwd())

    for key in treatments.keys():
        # Compute the target mean
        print("Working on {}".format(key))
        condition_data = [x for x in list(data[key]) if not np.isnan(x)]
        cond_avg = circ_mean(np.radians(condition_data))

        # Simulated summary stats
        treatment = treatments[key]

        #
        # Search for sufficiently low error
        #
        while True:
            simulator.simulate_treatment(treatment)
            changes = treatment.get_changes_in_bearing()
            deg_changes = np.degrees(changes)
            rounded_changes = [ int(5 * round(x/5)) for x in deg_changes ]
            sim_avg = circ_mean(np.radians(rounded_changes))

            error = polar_euclidean_distance(cond_avg, sim_avg)

            # Error threshold chosen to be sufficiently similar
            if error < 0.02:
                break

        #
        # Write out to dataframe
        #
        df = pd.DataFrame()
        df["changes"] = changes
        df["deg_changes"] = deg_changes
        df["rounded"] = rounded_changes
        df["modulo"] = [x % 360 for x in rounded_changes ]
        df.to_csv(
            "{}.csv".format(key)
        )
    return False

if __name__ == "__main__":
    # Simulator in use, best according to likelihood analysis
    sim = BWS(adjustment_slope=53,
              no_window=True,
              bias_variance=0.000303)

    #
    # Generate for either set of data. Comment as appropriate
    #
#    cue_weight_conditions(sim)
    sixty_multi_day_conditions(sim)
