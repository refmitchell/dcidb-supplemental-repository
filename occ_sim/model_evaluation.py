"""
model_evaluation.py

This script is used to perform likelihood-based evaluation of the different
populations generated using population_generation.py. There are a collection
of evaluation procedures which were used over the course of development; some
are legacy but they all follow the same basic structure. The parent population
distributions are read in along with the behavioural data, each behavioural
condition is then evaluated against the respective parent distribution by
using the parent distribution as a probability mass function (see the paper,
or respective function, for details).

Those marked [Legacy] were in use or useful at some point in development
but did not make it into the final paper, either due to an evolution of
a concept or because they did not add any new information.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import pdb

from util.integration_models import *

def slope_id(csv_name):
    """
    Extract an integer slope parameter from the filename.
    :param csv_name: The name of a csv file of the form NWS-<SLP>-0.5.csv
    :return: int(<SLP>)
    """
    return int(csv_name.split("-")[1])

def variance_id(csv_name):
    """
    Extract an integer slope parameter from the filename.
    :param csv_name: The name of a csv file of the form NWS-<SLP>-0.5.csv
    :return: int(<SLP>)
    """
    filename = csv_name.split("-")[2]
    return float(".".join([x for x in filename.split(".") if x != "csv"]))

def bws_variance_and_slope_evaluation():
    """
    [Legacy]
    Evaluate the Biased Weighted Sum model, changing both the variance
    and the slope at the same time. This result/analysis was not included as the
    result was extremely similar to the sequential analysis provided.
    """
    nbins = 72
    binwidth = 360 / nbins

    data = pd.read_csv("data/changes_full.csv")
    conditions = list(data.columns)

    files = [
        entry.name for entry in list(os.scandir("parent_distributions/var_and_slope_param_search"))
        if entry.name.split(".")[-1] == "csv"
    ]

    # Parameter sets
    variances = np.linspace(0, 0.0025, 25)
    slopes = np.arange(36, 61, 1)

    parent_distributions = [
        pd.read_csv("parent_distributions/var_and_slope_param_search/{}".format(filename))
        for filename in files
    ]

    # Divide each column by its sum
    norm_parent_distributions = [
        dist.div(dist.sum(axis=0), axis='columns') for dist in parent_distributions
    ]

    # Dictionary of parent distributions indexable by simulator.
    dist_dict = dict(zip(files, parent_distributions))
    norm_dist_dict = dict(zip(files, norm_parent_distributions))

    evaluation_df = pd.DataFrame(columns=conditions, index=norm_dist_dict.keys())
    best_possible_df = pd.DataFrame(columns=conditions, index=norm_dist_dict.keys())

    for model in evaluation_df.index:
        model_best = []
        for condition in evaluation_df.columns:
            # Probability using P(M|d) prop_to P(d|M)P(M)
            # Assuming even priors (poor assumption):
            # P(M|d) prop_to P(d|M)
            # All datapoints are assumed independent so
            # P(d|M) = p(d1|M) * P(d2|M) * ... * P(dN|M)

            # Data for this condition
            condition_data = list(data[condition])

            # Which bin does this datapoint belong to
            idxed_data = [ int(d / binwidth) for d in condition_data if not np.isnan(d) ]

            # Probability mass function for this condition and model
            model_pmf = norm_dist_dict[model][condition]

            data_probs = np.log(model_pmf.iloc[idxed_data])
            data_prob = data_probs.sum(axis='index')
            evaluation_df[condition][model] = data_prob


    evaluation_df["sum"] = evaluation_df.sum(axis='columns')

    mle = evaluation_df["sum"].max()
    evaluation_df["norm"] = evaluation_df["sum"].div(mle)

    pd.set_option("display.max_rows", None)

    likelihoods = np.empty((len(slopes), len(variances)))
    relative_likelihoods = np.empty((len(slopes), len(variances)))

    mle = evaluation_df["sum"].max()
    evaluation_df["norm"] = evaluation_df["sum"].div(mle)

    print(evaluation_df)
    print(evaluation_df[evaluation_df["norm"] == 1])

    evaluation_df.to_csv("narrow_bias_evaluation_df.csv")

    for i in range(len(slopes)):
        for j in range(len(variances)):
            idstr = "BWS-0.02-{}-{}.csv".format(slopes[i], variances[j])
            likelihoods[i][j] = evaluation_df.loc[idstr]["sum"]
            relative_likelihoods[i][j] = evaluation_df.loc[idstr]["norm"]

    fig = plt.figure(figsize=(12,7))
    ax = plt.subplot(111)
    vs, ws = np.meshgrid(variances,slopes)
    wmap = ax.pcolormesh(vs, ws, relative_likelihoods, shading='auto', cmap='viridis_r')
    cbar = fig.colorbar(wmap, ax=ax, shrink=0.6)

    plt.xlabel("$\sigma_{Bias}^2$ - Bias distribution variance")
    plt.ylabel("$a$ - slope parameter")

    plt.yticks(slopes)
    plt.xticks(variances, rotation='vertical')

    plt.title("Relative log likelihood for different $\sigma_{Bias}^2, a$ pairs")

    plt.savefig("../latex/img/var_and_slope_eval.png", bbox_inches="tight")
    plt.show()
    return evaluation_df


def bws_variance_evaluation():
    """
    Evaluate the Biased Weighted Sum model which excludes the concept of "bias
    windows" (i.e. only the variance of the bias distribution is changed).
    """

    nbins = 72
    binwidth = 360 / nbins

    data = pd.read_csv("data/changes_full.csv")
    conditions = list(data.columns)

    files = [
        entry.name for entry in list(os.scandir("parent_distributions/nowindow_bws_param_search"))
        if entry.name.split(".")[-1] == "csv"
    ]

    variances = np.linspace(0,0.01,100)
    parent_distributions = [
        pd.read_csv("parent_distributions/nowindow_bws_param_search/{}".format(filename))
        for filename in files
    ]

    # Divide each column by its sum
    norm_parent_distributions = [
        dist.div(dist.sum(axis=0), axis='columns') for dist in parent_distributions
    ]

    # Dictionary of parent distributions indexable by simulator.
    dist_dict = dict(zip(files, parent_distributions))
    norm_dist_dict = dict(zip(files, norm_parent_distributions))

    evaluation_df = pd.DataFrame(columns=conditions, index=norm_dist_dict.keys())
    best_possible_df = pd.DataFrame(columns=conditions, index=norm_dist_dict.keys())

    for model in evaluation_df.index:
        model_best = []
        for condition in evaluation_df.columns:
            # Probability using P(M|d) prop_to P(d|M)P(M)
            # Assuming even priors (poor assumption):
            # P(M|d) prop_to P(d|M)
            # All datapoints are assumed independent so
            # P(d|M) = p(d1|M) * P(d2|M) * ... * P(dN|M)

            # Data for this condition
            condition_data = list(data[condition])

            # Which bin does this datapoint belong to
            idxed_data = [ int(d / binwidth) for d in condition_data if not np.isnan(d) ]

            # Probability mass function for this condition and model
            model_pmf = norm_dist_dict[model][condition]

            data_probs = np.log(model_pmf.iloc[idxed_data])
            data_prob = data_probs.sum(axis='index')
            evaluation_df[condition][model] = data_prob


    evaluation_df["sum"] = evaluation_df.sum(axis='columns')

    mle = evaluation_df["sum"].max()
    evaluation_df["norm"] = evaluation_df["sum"].subtract(mle)
    print(evaluation_df[evaluation_df["norm"] == 1])
    pd.set_option("display.max_rows", None)
    print(evaluation_df)

    evaluation_df.to_csv("nowindow_bws_evaluation_df.csv")

    fig = plt.figure(figsize=(8,4.5))
    ax = plt.subplot(111)

    unsorted_ys = list(evaluation_df["norm"])
    unsorted_xs = [variance_id(x) for x in list(evaluation_df.index)]

    mle_var = evaluation_df[evaluation_df["sum"] == mle].index[0].split("-")[2][:-4]
    mle_var_float = float(mle_var)
    mle_var_str = "{:.06f}...".format(mle_var_float)
    print(mle_var_str)

    data = list(zip(unsorted_xs, unsorted_ys))
    data = sorted(data, key=lambda x: x[0])
    xs = [ x for (x,_) in data ]
    ys = [ y for (_,y) in data ]
    plt.xlim([0,0.01])
    # plt.ylim([0.99,1.01])
    plt.plot(xs, ys)
    plt.title("Log likelihood ratio w.r.t. variance MLE $\hat \sigma^2$")
    marker_lbl = r"$\hat{\sigma}^2 = " + mle_var_str + "$"
    print(marker_lbl)
    plt.axvline(mle_var_float,
                color="darkred",
                linestyle='dashed',
                label=marker_lbl)
    plt.xlabel("Variance parameter - $\sigma^2$")
    plt.ylabel("Log likelihood ratio, " + marker_lbl)
    plt.legend()
    plt.savefig("no_window_variance_analysis.svg")

    # plt.show()

    return True

def nws_param_evaluation():
    """
    Evaluate the Non-optimal Weighted Sum model across different potential
    adjustment slopes.
    """

    nbins = 72
    binwidth = 360 / nbins

    data = pd.read_csv("data/changes_full.csv")
    conditions = list(data.columns)

    files = [
        entry.name for entry in list(os.scandir("parent_distributions/nws_param_search"))
        if entry.name.split(".")[-1] == "csv"
    ]

    slopes = np.arange(0, 80, 1)
    biases = np.arange(0, 1, 0.01)

    parent_distributions = [
        pd.read_csv("parent_distributions/nws_param_search/{}".format(filename))
        for filename in files
    ]

    # Divide each column by its sum
    norm_parent_distributions = [
        dist.div(dist.sum(axis=0), axis='columns') for dist in parent_distributions
    ]

    # Dictionary of parent distributions indexable by simulator.
    dist_dict = dict(zip(files, parent_distributions))
    norm_dist_dict = dict(zip(files, norm_parent_distributions))

    evaluation_df = pd.DataFrame(columns=conditions, index=norm_dist_dict.keys())
    best_possible_df = pd.DataFrame(columns=conditions, index=norm_dist_dict.keys())

    for model in evaluation_df.index:
        model_best = []
        for condition in evaluation_df.columns:
            # Probability using P(M|d) prop_to P(d|M)P(M)
            # Assuming even priors (poor assumption):
            # P(M|d) prop_to P(d|M)
            # All datapoints are assumed independent so
            # P(d|M) = p(d1|M) * P(d2|M) * ... * P(dN|M)

            # Data for this condition
            condition_data = list(data[condition])


            # Which bin does this datapoint belong to
            idxed_data = [ int(d / binwidth) for d in condition_data if not np.isnan(d) ]

            # Probability mass function for this condition and model
            model_pmf = norm_dist_dict[model][condition]

            data_probs = np.log(model_pmf.iloc[idxed_data])
            data_prob = data_probs.sum(axis='index')
            evaluation_df[condition][model] = data_prob


    evaluation_df["sum"] = evaluation_df.sum(axis='columns')

    mle = evaluation_df["sum"].max()

    mle_slope = int(evaluation_df[evaluation_df["sum"] == mle].index[0].split("-")[1])
    evaluation_df["norm"] = evaluation_df["sum"].subtract(mle)
    print(evaluation_df[evaluation_df["norm"] == 1])
    pd.set_option("display.max_rows", None)
    print(evaluation_df)
    print(mle_slope)

    evaluation_df.to_csv("slope_evaluation_df.csv")

    fig = plt.figure(figsize=(8,4.5))
    ax = plt.subplot(111)

    unsorted_ys = list(evaluation_df["norm"])
    unsorted_xs = [slope_id(x) for x in list(evaluation_df.index)]

    data = list(zip(unsorted_xs, unsorted_ys))
    data = sorted(data, key=lambda x: x[0])
    xs = [ x for (x,_) in data ]
    ys = [ y for (_,y) in data ]
    plt.xlim([-1,80])
#    plt.ylim([0.99,1.03])
    plt.plot(xs, ys)
    plt.axvline(mle_slope, linestyle='--', color='darkred', label='$\hat{a} = 53$')
    plt.title("Log likelihood ratio w.r.t. slope MLE $\hat{a}$")
    plt.xlabel("Slope parameter - $a$")
    plt.ylabel(r'Log likelihood ratio, $\hat{a} = ' + str(mle_slope) + '$')
    plt.legend()
    plt.savefig("slope_likelihood_analysis.svg")
#    plt.show()

    return mle_slope

def cross_model_evaluation():
    """
    The final cross-model evaluation which checks each of the final candidate
    models against eachother. In this instance the AIC is also computed to
    provide a form of penalisation against the parameter counts.
    """

    nbins = 72
    binwidth = 360 / nbins

    data = pd.read_csv("data/changes_full.csv")

    conditions = list(data.columns)

    files = [
        entry.name for entry in list(os.scandir("parent_distributions/cross_model_eval"))
        if entry.name.split(".")[-1] == "csv"
    ]

    parent_distributions = [
        pd.read_csv("parent_distributions/cross_model_eval/{}".format(filename))
        for filename in files
    ]

    # Divide each column by its sum
    norm_parent_distributions = [
        dist.div(dist.sum(axis=0), axis='columns') for dist in parent_distributions
    ]

    # Dictionary of parent distributions indexable by simulator.
    dist_dict = dict(zip(files, parent_distributions))
    norm_dist_dict = dict(zip(files, norm_parent_distributions))

    evaluation_df = pd.DataFrame(columns=conditions, index=norm_dist_dict.keys())

    # Associate # of parameters for AIC computation to a filename key
    # AIC paramter counts are reused to compute the BIC
    aic_params = {k:None for k in evaluation_df.index}
    total_n = 0
    for k in list(aic_params.keys()):
        if "BWS" in k:
            aic_params[k] = 2
        elif "NWS" in k:
            aic_params[k] = 1
        else:
            aic_params[k] = 0

    for model in evaluation_df.index:
        total_n = 0 # All models use same data so we can take the last iteration
        for condition in evaluation_df.columns:
            # Probability using P(M|d) prop_to P(d|M)P(M)
            # Assuming even priors (poor assumption):
            # P(M|d) prop_to P(d|M)
            # All datapoints are assumed independent so
            # P(d|M) = p(d1|M) * P(d2|M) * ... * P(dN|M)

            # Data for this condition
            condition_data = list(data[condition])

            # Which bin does this datapoint belong to
            idxed_data = [ int(d / binwidth) for d in condition_data if not np.isnan(d) ]

            # Probability mass function for this condition and model
            model_pmf = norm_dist_dict[model][condition]
            data_probs = np.log(model_pmf.iloc[idxed_data])
            data_prob = data_probs.sum(axis='index')

            evaluation_df[condition][model] = data_prob
            total_n += len(idxed_data)


    evaluation_df["sum"] = evaluation_df.sum(axis='columns')
    print("Total n: {}".format(total_n))

    ml = evaluation_df["sum"].max()
    evaluation_df["norm"] = evaluation_df["sum"].subtract(ml)

    # Compute AIC and include in the evaluation_df
    # AIC = 2k - 2ln(Likelihood)
    # BIC = k*ln(total_n) - 2*ln(Likelihood)
    evaluation_df["aic"] = None
    evaluation_df["bic"] = None
    for m in aic_params.keys():
        p = aic_params[m]
        evaluation_df["aic"][m] = 2*p - 2*evaluation_df["sum"][m]
        evaluation_df["bic"][m] = p*np.log(total_n) - 2*evaluation_df["sum"][m]

    # Include relative measures in the result df; for aic/bic, lower is better
    evaluation_df["r_aic"] = evaluation_df["aic"].div(min(evaluation_df["aic"]))
    evaluation_df["r_bic"] = evaluation_df["bic"].div(min(evaluation_df["bic"]))
    evaluation_df.to_csv("cmedf.csv")

    fig = plt.figure(figsize=(12,7))
    ax = plt.subplot(111)
    evaluation_df["norm"].plot(ax=ax, kind='bar', color='k')
    plt.title("Log likelihood comparison across models")
    plt.xlabel("Model")
    plt.ylabel("Log likelihood ratio (lower is better)")
    labels = [x.split(".")[0] for x in aic_params]
    labels[0] = labels[0].split("-")[0]
    labels[1] = labels[1].split("-")[0]
    ax.set_xticklabels(labels)
    plt.ylim([-200, 5])

    fill_x = np.linspace(-10, 10, 100)
    fill_y = np.zeros(len(fill_x))
    plt.fill_between(fill_x, fill_y, color="grey")
    # plt.savefig("../latex/img/cme.png", bbox_inches="tight")
    # plt.show()
    return evaluation_df

def bws_param_evaluation():
    """
    [Legacy]
    An older version of the BWS parameter evaluation. This version included
    a concept of bias windows; regions within weight space where biases would
    have an effect. This was later found to be superfluous and somewhat
    inelegant, so the concept was dropped. BWS was then evaluated over the
    variance only.
    """
    nbins = 72
    binwidth = 360 / nbins

    data = pd.read_csv("data/changes_full.csv")
    conditions = list(data.columns)

    files = [
        entry.name for entry in list(os.scandir("parent_distributions/bws_param_search"))
        if entry.name.split(".")[-1] == "csv"
    ]

    variances = np.linspace(0, 0.01, 10)
    windows = np.linspace(0, 0.1, 10)

    parent_distributions = [
        pd.read_csv("parent_distributions/bws_param_search/{}".format(filename))
        for filename in files
    ]


    # Divide each column by its sum
    norm_parent_distributions = [
        dist.div(dist.sum(axis=0), axis='columns') for dist in parent_distributions
    ]

    # Dictionary of parent distributions indexable by simulator.
    dist_dict = dict(zip(files, parent_distributions))
    norm_dist_dict = dict(zip(files, norm_parent_distributions))

    evaluation_df = pd.DataFrame(columns=conditions, index=norm_dist_dict.keys())
    best_possible_df = pd.DataFrame(columns=conditions, index=norm_dist_dict.keys())

    for model in evaluation_df.index:
        for condition in evaluation_df.columns:
            # Probability using P(M|d) prop_to P(d|M)P(M)
            # Assuming even priors (poor assumption):
            # P(M|d) prop_to P(d|M)
            # All datapoints are assumed independent so
            # P(d|M) = p(d1|M) * P(d2|M) * ... * P(dN|M)

            # Data for this condition
            condition_data = list(data[condition])


            # Which bin does this datapoint belong to
            idxed_data = [ int(d / binwidth) % 72 for d in condition_data if not np.isnan(d) ]

            # Probability mass function for this condition and model
            model_pmf = norm_dist_dict[model][condition]

            data_probs = model_pmf.iloc[idxed_data]

            data_log_probs = np.log(data_probs)
            data_log_prob = data_log_probs.sum(axis='index')
            evaluation_df[condition][model] = data_log_prob


    evaluation_df["sum"] = evaluation_df.sum(axis='columns')

    likelihoods = np.empty((len(windows), len(variances)))
    relative_likelihoods = np.empty((len(windows), len(variances)))

    mle = evaluation_df["sum"].max()
    evaluation_df["norm"] = evaluation_df["sum"].div(mle)

    print(evaluation_df[evaluation_df["norm"] == 1])

    evaluation_df.to_csv("narrow_bias_evaluation_df.csv")

    for i in range(len(windows)):
        for j in range(len(variances)):
            idstr = "BWS-{}-{}.csv".format(windows[i], variances[j])

            likelihoods[i][j] = evaluation_df.loc[idstr]["sum"]
            relative_likelihoods[i][j] = evaluation_df.loc[idstr]["norm"]

    print(variances)
    print(windows)

    fig = plt.figure(figsize=(12,7))
    ax = plt.subplot(111)
    vs, ws = np.meshgrid(variances,windows)
    wmap = ax.pcolormesh(vs, ws, relative_likelihoods, shading='auto', cmap='viridis_r')
    cbar = fig.colorbar(wmap, ax=ax, shrink=0.6)

    plt.xlabel("$\sigma_{Bias}^2$ - Bias distribution variance")
    plt.ylabel("$\omega$ - Bias window")

    plt.yticks(windows)
    plt.xticks(variances)

    plt.title("Relative log likelihood for different $\sigma_{Bias}^2, \omega$ pairs")

    plt.savefig("../latex/img/narrow_bws_eval.png", bbox_inches="tight")
    plt.show()
    return evaluation_df


if __name__ == "__main__":
    """
    In development these were simply uncommented as necessary. They are
    intended to be run individually. For the same parent populations, the
    results should not change; if the parent populations are re-generated
    using population_generation.py, then the evaluation results may be slightly
    different. The overall picture however, should remain the same.
    """
    nws_param_evaluation()
    bws_variance_evaluation()
#    bws_param_evaluation()
#    bws_variance_and_slope_evaluation()
#    cross_model_evaluation()
#    print(nws_param_evaluation())
