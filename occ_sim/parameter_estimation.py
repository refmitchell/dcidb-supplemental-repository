
"""
parameter_estimation.py

[Legacy] - File now incompatible with the codebase

At an earlier stage in development; there were multiple sigmoid adjustment
functions in use. This script was used for parameter estimation for each
use case based on the existing data. The likelihood analysis for the NWS slope
replaces this.
"""

from util.treatment import Treatment
from util.models import ReliabilityModel
from util.integration_models import SingleCue
from util.integration_models import NoisyMurrayMorgansternCombinationSimulator
import util.models as r_model

from world.light import Light
from world.wind import Wind
from world.cue import Cue

import definitions as defn
import params

import matplotlib.pyplot as plt
from scipy.special import i0
import numpy as np
import os
import shutil

good_a = []
good_b = []

def sigmoid(x, slope, bias):
    return 1 / (1 + np.exp(-slope * x + bias))

def set_b_sigmoid(x, slope):
    bias = slope/2
    return sigmoid(x, slope, bias)

def kappa_adj_err_fn(speed, a, b):
    """
    :param a: the slope parameter of the adjustment function
    :param b: the bias parameter of the adjustment function
    """
    global good_a
    global good_b

    simulator = SingleCue(cue="wind")
    rel_model = ReliabilityModel()

    iterations = 100
    r_averages = []

    treatment = Treatment()

    treatment.set_reliability_model(rel_model)
    treatment.set_n(10) # Set to match the data I'm working from
    treatment.set_id("Empty")

    dummy_cue = Cue("light", 0, 0, treatment)
    test_cue = Wind(speed, np.radians(0), treatment)
    test_cue.set_r_adjustment(a, b)
    initial = [test_cue, dummy_cue]

    test_cue_resample = Wind(speed, np.radians(0), treatment)
    test_cue_resample.set_r_adjustment(a, b)
    resample = [test_cue_resample, dummy_cue]

    treatment.set_initial_cues(initial)

    # Not truly a conflict in this context, resampling
    # to get von Mises noise against which we can check
    # accuracy
    treatment.set_conflict_cues(resample)

    # Expected R should be the same for both cues as the
    # input elevation is identical.
    expected_r = test_cue.get_reliability()

    for n in range(iterations):
        simulator.simulate_treatment(treatment)

        # Temp for testing
        changes = treatment.get_changes_in_bearing()
        avg_r, avg_t = treatment.get_avg_change()
        r_averages.append(avg_r)

    overall_avg_r = sum(r_averages) / len(r_averages)
    error = (overall_avg_r - expected_r)**2

    if error < 0.005:
 #       print("Parameters: {} {}".format(a, b))
        good_a.append(a)
        good_b.append(b)

    return error

def kappa_adjustment_search():
    global good_a
    global good_b

    n = 50
    slopes = np.linspace(0,200,n)
    biases = np.linspace(0,200,n)

    errors = np.empty((len(slopes), len(biases)))

    elevations_deg =list(range(91))[1:]# Don't ask

    idx = 0
    elevations = np.radians(elevations_deg)
    wind_speeds = np.linspace(0.5, 2.5, 90)
    fig = plt.figure(figsize=(16,9))
    axes = []
    total = n**2 * len(elevations)

    iter = 1
    with open("r_adj_parameters_wind.txt", 'w') as f:
        for speed in wind_speeds:
            good_a = []
            good_b = []
            for slope in range(len(slopes)):
                for bias in range(len(biases)):
                    print("Speed: {:.02f}, Iteration: {}/{}"
                          .format(speed,
                                  iter,
                                  total)
                    )
                    errors[bias, slope] = kappa_adj_err_fn(speed,
                                                           slopes[slope],
                                                           biases[bias])
                    iter += 1
            good_params = list(zip(good_a,good_b))
            f.write("Speed: {:.02f}, Params: {}\n".format(speed,
                                                          good_params)
            )
            ax = fig.add_subplot(9, 10, idx + 1)
            ax.set_aspect(1)
            plt.xlim([0,200])
            plt.ylim([0,200])
            wmap = ax.pcolormesh(slopes,
                                 biases,
                                 errors,
                                 shading='auto',
                                 cmap='binary')
            ax.plot(good_a,
                    good_b,
                    'ro',
                    markersize=0.5,
                    label="Squared error < 0.005")

            axes.append(ax)
            idx +=1

    plt.savefig("r_adjustment_parameter_space_full.png", bbox_inches="tight")

def individual_noise_err_fn(treatment,
                            result,
                            noise_mean,
                            noise_variance,
                            iterations):
    simulator = NoisyMurrayMorgansternCombinationSimulator(bias_mu=noise_mean,
                                                           bias_variance=noise_variance)

    Rs = []
    thetas = []
    for i in range(iterations):
        simulator.simulate_treatment(treatment)
        R, theta = treatment.get_avg_change()
        Rs.append(R)
        thetas.append(theta)


    # Convert to cartesian
    observed = [
        (r * np.cos(theta), r * np.sin(theta)) for (r, theta) in zip(Rs, thetas)
    ]

    # Take average x, y; This is the average response over iterations
    obs_x = sum([x for (x,_) in observed]) / len(observed)
    obs_y = sum([y for (_,y) in observed]) / len(observed)

    # Convert expected to cartesian
    exp_R, exp_T = result[0], result[1]
    exp_x = exp_R * np.cos(exp_T)
    exp_y = exp_R * np.sin(exp_T)

    ## Continue here; take error metric of your choice (there's a few to choose
    ## from). Could take the mean and check the error or check the average error.
    errs = []
    for x, y in observed:
        # Error metric, mean sum of absolute error
        err = (abs(x - exp_x) +  abs(y - exp_y)) / 2
        errs.append(err)

#    average sum of absolute errors
    error = sum(errs) / len(errs)

    #error = (abs(obs_x - exp_x) + abs(obs_y - exp_y)) / 2
    return error

def individual_noise_search():

    # Individual noise parameters
    n = 50
    means = np.linspace(-0.3,0.3,n)
    variances = np.linspace(0.0001, 0.6, n)
    iterations = 50

    rel_model = ReliabilityModel()

    # The tweleve treatments we want to simulate
    elevations = [45, 60, 75, 86]
    conflicts = [0, 60, 120]

    rvals = [0.89, 0.76, 0.50, 0.88, 0.42, 0.08, 0.74, 0.61, 0.57, 0.73, 0.38, 0.52]
    thetas = [357, 351, 16, 349, 55, 142, 357, 78, 116, 0, 74, 123]
    outputs = list(zip(rvals, np.radians(thetas)))
    treatments = []

    for elevation in elevations:
        for conflict in conflicts:
            treatment = Treatment()
            treatment.set_reliability_model(rel_model)
            treatment.set_n(30)
            treatment.set_id("Elevation: {}, Conflict: {}".format(elevation, conflict))

            init_light = Light(np.radians(elevation), np.radians(0), treatment)
            init_wind = Wind(2.5, np.radians(0), treatment)
            initial = [init_wind, init_light]

            conf_light = Light(np.radians(elevation), np.radians(0), treatment)
            conf_wind = Wind(2.5, np.radians(conflict), treatment)
            conflict = [conf_wind, conf_light]

            treatment.set_initial_cues(initial)
            treatment.set_conflict_cues(conflict)
            treatments.append(treatment)


    fig = plt.figure(figsize=(16,9))
    idx = 0
    errors = np.empty((len(means), len(variances)))
    cum_errors = np.zeros((len(means), len(variances)))
    total = n**2

    with open("individual_noise_parameters.txt", 'w') as f:
        while idx < len(treatments):
            treatment = treatments[idx]
            result = outputs[idx]
            print(result)
            iter = 1
            for mean_idx in range(len(means)):
                for variance_idx in range(len(variances)):
                    print("{}, Iteration: {}/{}"
                          .format(treatment.get_id(),
                                  iter,
                                  total)
                    )
                    errors[variance_idx, mean_idx] = individual_noise_err_fn(treatment,
                                                                             result,
                                                                             means[mean_idx],
                                                                             variances[variance_idx],
                                                                             iterations
                    )

                    iter += 1

            # Update Cumulative error
            cum_errors += errors

            ax = fig.add_subplot(1, 1, 1)
            ax.set_aspect(1)
            plt.xlim([-0.3, 0.3])
            plt.ylim([0.0001, 0.6])
            wmap = ax.pcolormesh(means,
                                 variances,
                                 errors,
                                 shading='auto',
                                 cmap='binary')
            cbar = plt.colorbar(wmap, shrink=0.5, aspect=5)
            ax.set_title(treatment.get_id())
            ax.set_xlabel("Bias")
            ax.set_ylabel("Variance")
            plt.savefig("{:05d}.png".format(idx), bbox_inches="tight")
            plt.clf()
            idx +=1

        # Create a final plot of cumulative error over all conditions.
        #cum_errors = cum_errors / sum(sum(cum_errors))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect(1)
        plt.xlim([-0.3, 0.3])
        plt.ylim([0.0001, 0.6])
        wmap = ax.pcolormesh(means,
                             variances,
                             cum_errors,
                             shading='auto',
                             cmap='binary')
        cbar = plt.colorbar(wmap, shrink=0.5, aspect=5)
        ax.set_title("Cumulative error")
        ax.set_xlabel("Bias")
        ax.set_ylabel("Variance")
        plt.savefig("cumulative_errors.png", bbox_inches="tight")

    return False

# Run a given parameter search
if __name__ == '__main__':
    individual_noise_search()




