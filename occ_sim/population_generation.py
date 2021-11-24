
"""
population_generation.py

Generate "parent" populations for each set of conditions and each
simulation method. This script will create a file for each simulator,
with distributions for each condition (twelve in total).

Heavy code dupliaction as some methods have been kept separate for
ease in data management and repeated runs.
"""

from util.treatment import Treatment
from util.models import ReliabilityModel
from util.integration_models import *
from util.pmmcs_model import BiasedWeightedSum

import model_evaluation

from world.wind import Wind
from world.light import Light

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import smtplib

def variance_only_bws_param_generation(slope=53):
    """
    Generate populations varying only the variance parameter of the
    BWS. Bias windows are totally ignored (bias is always applied).

    Given a slope parameter, we want to find
    maximally likely bias parameters.
    """

    # Generate all experimental conditions
    elevations = [45, 60, 75, 86]
    conflicts = [0, 60, 120]
    windspeeds = [1.25, 2.5]
    treatments = dict()

    # Parameter sets
    variances = np.linspace(0, 0.01, 100)
    simulators = []

    for v in variances:
        simulators.append(BWS(bias_variance=v,
                              adjustment_slope=slope,
                              no_window=True
        ))

    rel_model = ReliabilityModel()

    os.chdir("parent_distributions/nowindow_bws_param_search")
    print("Slope parameter initialised as {}, CHECK THIS IS CORRECT!".format(slope))

    # Put into dataframes; one column for each condition, one file for each
    # simulator.
    print(os.getcwd())

    for wind_speed in windspeeds:
        for elevation in elevations:
            # Condition doesn't exist so don't generate unnecessary work.
            if elevation == 45 and wind_speed == 1.25:
                print("Skipping condition {}-{}".format(
                    wind_speed, elevation
                ))
                continue
            for conflict in conflicts:
                # Generate treatment for each condition with extremely
                # large sample sizes to get a better picture of the
                # parent population.
                treatment = Treatment()
                treatment.set_reliability_model(rel_model)
                treatment.set_n(1000000)
                treatment.set_id("E:{};C:{}".format(elevation, conflict))

                init_light = Light(np.radians(elevation), np.radians(0), treatment)
                init_wind = Wind(wind_speed, np.radians(0), treatment)
                initial = [init_wind, init_light]

                conf_light = Light(np.radians(elevation), np.radians(0), treatment)
                conf_wind = Wind(wind_speed, np.radians(conflict), treatment)
                conf= [conf_wind, conf_light]

                treatment.set_initial_cues(initial)
                treatment.set_conflict_cues(conf)
                treatment.generate_samples()

                key = "{:.02f}-{}-{:03d}".format(wind_speed, elevation, conflict)
                treatments[key] = treatment

    print("Treatment generation complete")

    for sim in simulators:
        dataframe = pd.DataFrame()
        for key in treatments.keys():

            treatment = treatments[key]

            # Run simulation
            start_time = time.time()
            sim.simulate_treatment(treatment)
            runtime = time.time() - start_time

            print("{}-{} - Runtime: {}".format(sim.name(),key,runtime))

            # Bearing changes in radians
            changes = treatment.get_changes_in_bearing()

            # List histogram, 5 degree bins from 0-355.
            # (Normalisation can be done during analysis)
            nbins = 72
            ch_hist = np.histogram(np.degrees(changes) % 360,
                                   np.linspace(0, 360, nbins + 1)
            )[0]

            dataframe[key] = ch_hist

        dataframe.to_csv("{}.csv".format(sim.name()), index=False)

    return False

def variance_and_slope_bws_param_generation():
    """
    [Legacy]
    Generate populations for a subset of parameterisations of the BWS model.
    The parameters varied affect the biases in use (bias window and variance).

    Given a slope parameter, we want to find maximally likely bias parameters.

    This was run along with an evaluation routine, however the results did not
    significantly change our conclusions. The 'optimal' parameter returned were
    slightly different, however there was not enough of a difference to warrant
    further work given the computational complexity of this routine.
    """

    # Generate all experimental conditions
    elevations = [45, 60, 75, 86]
    conflicts = [0, 60, 120]
    windspeeds = [1.25, 2.5]
    treatments = dict()

    # Parameter sets
    variances = np.linspace(0, 0.0025, 25)
    slopes = np.arange(36, 61, 1)

    simulators = []

    for v in variances:
        for slope in slopes:
            simulators.append(BWS(bias_variance=v,
                                  adjustment_slope=slope,
                                  no_window=True
            ))

    rel_model = ReliabilityModel()

    os.chdir("parent_distributions/var_and_slope_param_search")
    print("Slope parameter initialised as {}, CHECK THIS IS CORRECT!".format(slope))

    # Put into dataframes; one column for each condition, one file for each
    # simulator.
    print(os.getcwd())

    for wind_speed in windspeeds:
        for elevation in elevations:
            # Condition doesn't exist so don't generate unnecessary work.
            if elevation == 45 and wind_speed == 1.25:
                print("Skipping condition {}-{}".format(
                    wind_speed, elevation
                ))
                continue
            for conflict in conflicts:
                # Generate treatment for each condition with extremely
                # large sample sizes to get a better picture of the
                # parent population.
                treatment = Treatment()
                treatment.set_reliability_model(rel_model)
                treatment.set_n(1000000)
                treatment.set_id("E:{};C:{}".format(elevation, conflict))

                init_light = Light(np.radians(elevation), np.radians(0), treatment)
                init_wind = Wind(wind_speed, np.radians(0), treatment)
                initial = [init_wind, init_light]

                conf_light = Light(np.radians(elevation), np.radians(0), treatment)
                conf_wind = Wind(wind_speed, np.radians(conflict), treatment)
                conf= [conf_wind, conf_light]

                treatment.set_initial_cues(initial)
                treatment.set_conflict_cues(conf)
                treatment.generate_samples()

                key = "{:.02f}-{}-{:03d}".format(wind_speed, elevation, conflict)
                treatments[key] = treatment

    print("Treatment generation complete")

    for sim in simulators:
        dataframe = pd.DataFrame()
        for key in treatments.keys():

            treatment = treatments[key]

            # Run simulation
            start_time = time.time()
            sim.simulate_treatment(treatment)
            runtime = time.time() - start_time

            print("{}-{} - Runtime: {}".format(sim.name(),key,runtime))

            # Bearing changes in radians
            changes = treatment.get_changes_in_bearing()

            # List histogram, 5 degree bins from 0-355.
            # (Normalisation can be done during analysis)
            nbins = 72
            ch_hist = np.histogram(np.degrees(changes) % 360,
                                   np.linspace(0, 360, nbins + 1)
            )[0]

            dataframe[key] = ch_hist

        dataframe.to_csv("{}.csv".format(sim.name()), index=False)

    return False

def bws_param_generation(slope=53):
    """
    Generate populations for a subset of parameterisations of the BWS model.
    The parameters varied affect the biases in use (bias window and variance).

    Given a slope parameter, we want to find maximally likely bias parameters.

    :param slope: The slope parameter, found as most likely for NWS.
    """
    # Generate all 12 experimental conditions
    elevations = [45, 60, 75, 86]
    conflicts = [0, 60, 120]
    windspeeds = [1.25, 2.5]
    treatments = dict()

    # Parameter sets
    variances = np.linspace(0, 0.001, 10)
    windows = np.linspace(0, 0.01, 10)

    simulators = []

    for v in variances:
        for w in windows:
            simulators.append(BWS(bias_window=w,
                                  bias_variance=v,
                                  adjustment_slope=slope))

    rel_model = ReliabilityModel()

    os.chdir("parent_distributions/narrow_bws_param_search")
    print("Slope parameter initialised as {}, CHECK THIS IS CORRECT!".format(slope))
    input("Working directory is {}; press any key to continue...".format(os.getcwd()))

    # Put into dataframes; one column for each condition, one file for each
    # simulator.
    print(os.getcwd())

    for wind_speed in windspeeds:
        for elevation in elevations:
            # Condition doesn't exist so don't generate unnecessary work.
            if elevation == 45 and wind_speed == 1.25:
                print("Skipping condition {}-{}".format(
                    wind_speed, elevation
                ))
                continue
            for conflict in conflicts:
                # Generate treatment for each condition with extremely
                # large sample sizes to get a better picture of the
                # parent population.
                treatment = Treatment()
                treatment.set_reliability_model(rel_model)
                treatment.set_n(1000000)
                treatment.set_id("E:{};C:{}".format(elevation, conflict))

                init_light = Light(np.radians(elevation), np.radians(0), treatment)
                init_wind = Wind(wind_speed, np.radians(0), treatment)
                initial = [init_wind, init_light]

                conf_light = Light(np.radians(elevation), np.radians(0), treatment)
                conf_wind = Wind(wind_speed, np.radians(conflict), treatment)
                conf= [conf_wind, conf_light]

                treatment.set_initial_cues(initial)
                treatment.set_conflict_cues(conf)
                treatment.generate_samples()

                key = "{:.02f}-{}-{:03d}".format(wind_speed, elevation, conflict)
                treatments[key] = treatment

    print("Treatment generation complete")

    for sim in simulators:
        dataframe = pd.DataFrame()
        for key in treatments.keys():

            treatment = treatments[key]

            # Run simulation
            start_time = time.time()
            sim.simulate_treatment(treatment)
            runtime = time.time() - start_time

            print("{}-{} - Runtime: {}".format(sim.name(),key,runtime))

            # Bearing changes in radians
            changes = treatment.get_changes_in_bearing()

            # List histogram, 5 degree bins from 0-355.
            # (Normalisation can be done during analysis)
            nbins = 72
            ch_hist = np.histogram(np.degrees(changes) % 360,
                                   np.linspace(0, 360, nbins + 1)
            )[0]

            dataframe[key] = ch_hist

        dataframe.to_csv("{}.csv".format(sim.name()), index=False)

    return False

def nws_param_generation():
    """
    Generate populations for a subset of parmeterisations of the NWS
    model.  The parameters varied affect the slope parameter of the
    adjustment function the goal is to see how likelihood changes from
    no slope to extreme (pseudo-WTA) slope.

    Note that NWS is a sub-model of BWS so we use BWS to avoid repeating code.
    NWS is BWS without biases.

    We want to find a maximally likely slope parameter.
    """
    # Generate all 12 experimental conditions
    elevations = [45, 60, 75, 86]
    conflicts = [0, 60, 120]
    windspeeds = [1.25, 2.5]
    treatments = dict()

    simulators = []

    # Investigate integer slopes from 0 (linear) to 100
    adjustment_slopes = np.arange(0,100,1)

    for s in adjustment_slopes:
        simulators.append(BWS(bias_window=-1,
                              adjustment_slope=s,
                              adjustment_bias=0.5))

    rel_model = ReliabilityModel()

    os.chdir("parent_distributions/nws_param_search")
#    input("Working directiory is {}; press any key to continue.".format(os.getcwd()))

    # Put into dataframes; one column for each condition, one file for each
    # simulator.
    print(os.getcwd())

    for wind_speed in windspeeds:
        for elevation in elevations:
            for conflict in conflicts:
                # Generate treatment for each condition with extremely
                # large sample sizes to get a better picture of the
                # parent population.
                treatment = Treatment()
                treatment.set_reliability_model(rel_model)
                treatment.set_n(1000000)
                treatment.set_id("E:{};C:{}".format(elevation, conflict))

                init_light = Light(np.radians(elevation), np.radians(0), treatment)
                init_wind = Wind(wind_speed, np.radians(0), treatment)
                initial = [init_wind, init_light]

                conf_light = Light(np.radians(elevation), np.radians(0), treatment)
                conf_wind = Wind(wind_speed, np.radians(conflict), treatment)
                conf = [conf_wind, conf_light]

                treatment.set_initial_cues(initial)
                treatment.set_conflict_cues(conf)
                treatment.generate_samples()

                key = "{:.02f}-{}-{:03d}".format(wind_speed, elevation, conflict)
                treatments[key] = treatment

    print("Treatment generation complete")

    for sim in simulators:
        dataframe = pd.DataFrame()
        for key in treatments.keys():
            print("{}-{}".format(sim.name(),key))
            treatment = treatments[key]

            # Run simulation
            sim.simulate_treatment(treatment)

            # Bearing changes in radians
            changes = treatment.get_changes_in_bearing()

            # List histogram, 5 degree bins from 0-355.
            # (Normalisation can be done during analysis)
            nbins = 72
            ch_hist = np.histogram(np.degrees(changes) % 360,
                                   np.linspace(0, 360, nbins + 1)
            )[0]

            dataframe[key] = ch_hist

        dataframe.to_csv("{}.csv".format(sim.name()), index=False)

    return False

def cross_model_population_generation():
    """
    Generate populations for all models (WTA, WAM, CMLE, NWS, and BWS)
    and compare (1) the likelihoods, and (2) the AIC result.
    """
    elevations = [45, 60, 75, 86]
    conflicts = [0, 60, 120]
    windspeeds = [1.25, 2.5]
    treatments = dict()

    # Default parameters where required are set to those which best fit the data.
    simulators = [WTA(), # Winner takes all
                  WAM(), # Weighted arithmetic nean
                  CMLE(), # Circular Maximum Likelihood Estimate
                  BWS(bias_window=-1, adjustment_slope=53), # Non-optimal weighted sum
                  BWS(adjustment_slope=53, bias_variance=0.000303, no_window=True) # Biased with no window
    ]

    rel_model = ReliabilityModel()

    os.chdir("parent_distributions/cross_model_eval")
    input("Working directiory is {}; press any key to continue.".format(os.getcwd()))

    # Put into dataframes; one column for each condition, one file for each
    # simulator.
    for wind_speed in windspeeds:
        for elevation in elevations:
            if wind_speed == 1.25 and elevation == 45:
                continue
            for conflict in conflicts:
                # Generate treatment for each condition with extremely
                # large sample sizes to get a better picture of the
                # parent population.
                treatment = Treatment()
                treatment.set_reliability_model(rel_model)
                treatment.set_n(1000000)
                treatment.set_id("E:{};C:{}".format(elevation, conflict))

                init_light = Light(np.radians(elevation), np.radians(0), treatment)
                init_wind = Wind(wind_speed, np.radians(0), treatment)
                initial = [init_wind, init_light]

                conf_light = Light(np.radians(elevation), np.radians(0), treatment)
                conf_wind = Wind(wind_speed, np.radians(conflict), treatment)
                conf = [conf_wind, conf_light]

                treatment.set_initial_cues(initial)
                treatment.set_conflict_cues(conf)
                treatment.generate_samples()

                key = "{:.02f}-{}-{:03d}".format(wind_speed, elevation, conflict)
                treatments[key] = treatment

    print("Treatment generation complete")

    for sim in simulators:
        dataframe = pd.DataFrame()
        for key in treatments.keys():
            treatment = treatments[key]

            # Run simulation
            sim.simulate_treatment(treatment)

            # Bearing changes in radians
            changes = treatment.get_changes_in_bearing()

            # List histogram, 5 degree bins from 0-355.
            # (Normalisation can be done during analysis)
            nbins = 72
            ch_hist = np.histogram(np.degrees(changes) % 360,
                                   np.linspace(0, 360, nbins + 1)
            )[0]

            dataframe[key] = ch_hist

        dataframe.to_csv("{}.csv".format(sim.name()), index=False)

    return False

def sendmail_on_completion(text=""):
    """
    Send mail on population generation completion. Useful for long generations
    run on remote systems.

    :param text: Text to be included in the email.
    """
    sender = "r.mitchell@ed.ac.uk"
    receiver = ["r.mitchell@ed.ac.uk"]

    headers = "From: {}\nTo:{}\nSubject:Population generation complete".format(sender, receiver[0])
    message = """{}\nPopulation generation complete for {}, results ready on DICE.""".format(headers, text)

    server = smtplib.SMTP('localhost')
    server.sendmail(sender, receiver, message)
    server.quit()

if __name__ == "__main__":
    """
    Designed to be uncommented as necessary. These must be run individually due
    to directory switching.
    """
#    nws_param_generation()
#    variance_only_bws_param_generation()
    variance_and_slope_bws_param_generation()
#    cross_model_population_generation()
    # Send an email to notify that the process is complete
    sendmail_on_completion("var and slope")









