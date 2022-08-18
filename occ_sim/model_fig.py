"""
model_fig.py

Includes utilities and plotting code to produce the model
comparison figure. The figure serves as an illustrative comparison
of differing model predictions at near-equal weights.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import vonmises as np_vonmises
from scipy.special import i0
from scipy.stats import norm


def vonmises_pdf(x, mu, kappa):
    return np.exp(kappa*np.cos(x - mu))/(2*np.pi * i0(kappa))

def sigmoid(x, slope=53, bias=0.5):
    """
    Sigmoid adjustment with the purpose of inflating values above 0.5 and
    suppressing those below.

    :param x: input value
    :param slope: the slope of the sigmoid (how steep it is)
    :param bias: the horizontal shift (should always be 0.5)
    :return: The adjusted value.
    """
    return 1 / (1 + np.exp(-slope *(x - bias)))

def vec_sum(a, b):
    """
    Add two cartesian vectors together.

    :param a: lhs
    :param b: rhs
    :return: The cartesian result
    """
    return [a[0] + b[0], a[1] + b[1]]

def cartesian(a):
    """
    Convert a polar vector of the form (r, theta) to cartesian (x, y)
    co-ordinates.
    :param a: A polar vector of the form (r, theta) (np array or list,
              subscriptable and mutable)
    :return: Cartesian vector (x,y)
    """
    r = a[0]
    t = a[1]
    return [r*np.cos(t),r*np.sin(t)]

def polar(a):
    """
    Convert a cartesian vector of the form (x,y) to polar (r, theta)
    co-ordinates.
    :param a: A cartesian vector of the form (x,y) (np array or list,
              subscriptable and mutable)
    :return: Polar vector (r, theta)
    """
    x = a[0]
    y = a[1]
    return [np.sqrt(x**2 + y**2), np.arctan2(y, x)]

def polar_vector_sum(a, b):
    u = cartesian(a)
    v = cartesian(b)
    r = polar(vec_sum(u, v))
    return r

def winner_take_all(a,b):
    return a if a[0] > b[0] else b

def weighted_arithmetic_mean(a,b):
    weights = [a[0], b[0]]
    angles = [a[1], b[1]]
    mean = wavg(weights, angles)
    r = sum(weights)
    return [r, mean]

def wavg(weights, values):
    a = [ weight * value for (weight,value) in zip(weights,values) ]
    return sum(a)/len(a)

def nonoptimal_vector_sum(a,b):
    adj_a = [sigmoid(a[0]), a[1]]
    adj_b = [sigmoid(b[0]), b[1]]
    return polar_vector_sum(adj_a, adj_b)

def biased_vector_sum(a,b):
    bias = norm.rvs(0, np.sqrt(0.000303))
    a_cpy = [sigmoid(a[0] + bias), a[1]]
    b_cpy = [sigmoid(b[0] - bias), b[1]]
    return polar_vector_sum(a_cpy, b_cpy)

def mean_vector(a):
    """
    Compute mean vector from a list of angles
    """
    polars = [[1, x] for x in a]
    carts = [cartesian(x) for x in polars]
    xs = [ x for [x, _] in carts ]
    ys = [ y for [_, y] in carts ]
    avg_x = sum(xs)/len(xs)
    avg_y = sum(ys)/len(ys)
    return polar([avg_x, avg_y])

def generate_dataset(a1, a2, k1, k2, n):
    W1 = k1 / (k1 + k2)
    W2 = k2 / (k1 + k2)

    c1 = np.array([W1, a1])
    c2 = np.array([W2, a2])

    c1_samples = np_vonmises(a1, k1, n)
    c2_samples = np_vonmises(a2, k2, n)

    c1_vecs = [[W1, sample] for sample in c1_samples]
    c2_vecs = [[W2, sample] for sample in c2_samples]

    # Vector manipulation - for each sample do the calculation
    polar_sum = []
    wta = []
    wam = []
    nvs = []
    bvs = []
    for (sample1, sample2) in zip(c1_vecs, c2_vecs):
        polar_sum.append(polar_vector_sum(sample1,sample2))
        wta.append(winner_take_all(sample1, sample2))
        wam.append(weighted_arithmetic_mean(sample1, sample2))
        nvs.append(nonoptimal_vector_sum(sample1, sample2))
        bvs.append(biased_vector_sum(sample1, sample2))

    results = dict()
    results["c1_samples"] = c1_samples
    results["c2_samples"] = c2_samples
    results["wvs"] = polar_sum
    results["wta"] = wta
    results["wam"] = wam
    results["nvs"] = nvs
    results["bvs"] = bvs
    return results

if __name__ == "__main__":
    k1 = 2
    k2 = 2.05
    n = 1000
    conflicts = np.radians([0,60,120])
    # Data as dictionaries for each conflict [ 0, 60, 120 ]
    full_results = []
    for conflict in conflicts:
        full_results.append(generate_dataset(0, conflict, k1, k2, n))

    fig, axs = plt.subplots(nrows=6,ncols=len(conflicts+1),sharex=True,sharey=True)
    fig.set_size_inches(8.25, 8.25)
    input_axs = axs[0]
    bvs_axs = axs[1]
    nvs_axs = axs[2]
    wvs_axs = axs[3]
    wta_axs = axs[4]
    wam_axs = axs[5]

    inset_axs = []

    # For each column
    for i in range(len(conflicts)):
        results = full_results[i]
        c1_samples = results["c1_samples"]
        c2_samples = results["c2_samples"]
        bvs = [ th for [_, th] in results["bvs"]]
        nvs = [ th for [_, th] in results["nvs"]]
        wvs = [ th for [_, th] in results["wvs"]]
        wta = [ th for [_, th] in results["wta"]]
        wam = [ th for [_, th] in results["wam"]]

        bvscol = 'tab:red'
        nvscol = 'tab:orange'
        wvscol = 'tab:grey'
        wtacol = 'tab:purple'
        wamcol = 'deeppink'

        d = False
        meanwidth = 0.8
        ran = (-np.pi, np.pi)
        nbins=72

        # Inputs
        col1 = 'tab:green'
        col2 = 'tab:blue'
        ths = np.linspace(-np.pi, np.pi, 1000)
        y1 = vonmises_pdf(ths, 0, k1)
        y2 = vonmises_pdf(ths, conflicts[i], k2)
        c1_r, c1_mean = mean_vector(c1_samples)
        c2_r, c2_mean = mean_vector(c2_samples)
        input_axs[i].hist(c1_samples, nbins, range=ran, color=col1, alpha=0.5, density=False, label="Cue 1 sample density")
        input_axs[i].hist(c2_samples, nbins, range=ran, color=col2, alpha=0.5, density=False, label="Cue 2 sample density")
        # input_axs[i].plot(ths, y1, color=col1, label="Cue 1 noise distribution")
        # input_axs[i].plot(ths, y2, color=col2, label="Cue 2 noise distribution")
        input_axs[i].axvline(x=c1_mean, color=col1, linestyle='dotted', linewidth=meanwidth)
        input_axs[i].axvline(x=c2_mean, color=col2, linestyle='dashed', linewidth=meanwidth)

        input_axs[i].text(0.99,
                          0.8,
                          "{}$^\circ$".format(int(np.ceil(np.degrees(conflicts[i])))),
                          transform=input_axs[i].transAxes,
                          fontsize="medium",
                          horizontalalignment="right"
        )

        bvs_r, bvs_mean = mean_vector(nvs)
        bvs_axs[i].hist(bvs, nbins, range=ran, color=bvscol, alpha=0.5, density=d, label="BVS")
        bvs_axs[i].axvline(x=bvs_mean, color=bvscol, linestyle='dashed', linewidth=meanwidth)
        bvs_mean_axs = bvs_axs[i].twinx()
        bvs_mean_axs.set_ylim([0, 1.2])
        bvs_mean_axs.axhline(y=bvs_r, color=bvscol, linestyle='dashed', linewidth=meanwidth)


        nvs_r, nvs_mean = mean_vector(nvs)
        nvs_axs[i].hist(nvs, nbins, range=ran, color=nvscol, alpha=0.3, density=d)
        nvs_axs[i].axvline(x=nvs_mean, color=nvscol, linestyle='dashed', linewidth=meanwidth)
        nvs_mean_axs = nvs_axs[i].twinx()
        nvs_mean_axs.set_ylim([0, 1.2])
        nvs_mean_axs.axhline(y=nvs_r, color=nvscol, linestyle='dashed', linewidth=meanwidth)


        wvs_r, wvs_mean = mean_vector(wvs)
        wvs_axs[i].hist(wvs, nbins, range=ran, color=wvscol, alpha=0.5, density=d)
        wvs_axs[i].axvline(x=wvs_mean, color=wvscol, linestyle='dashed', linewidth=meanwidth)
        wvs_mean_axs = wvs_axs[i].twinx()
        wvs_mean_axs.set_ylim([0, 1.2])
        wvs_mean_axs.axhline(y=wvs_r, color=wvscol, linestyle='dashed', linewidth=meanwidth)


        wta_r, wta_mean = mean_vector(wta)
        wta_axs[i].hist(wta, nbins, range=ran, color=wtacol, alpha=0.5, density=d)
        wta_axs[i].axvline(x=wta_mean, color=wtacol, linestyle='dashed', linewidth=meanwidth)
        wta_mean_axs = wta_axs[i].twinx()
        wta_mean_axs.set_ylim([0, 1.2])
        wta_mean_axs.axhline(y=wta_r, color=wtacol, linestyle='dashed', linewidth=meanwidth)


        wam_r, wam_mean = mean_vector(wam)
        wam_axs[i].hist(wam, bins=nbins, range=ran, color=wamcol, alpha=0.5, density=d)
        wam_axs[i].axvline(x=wam_mean, color=wamcol, linestyle='dashed', linewidth=meanwidth)
        wam_axs[i].set_xticks([-np.pi, 0, np.pi])
        wam_axs[i].set_xticklabels(['-180', '0', '180'])
        wam_mean_axs = wam_axs[i].twinx()
        wam_mean_axs.set_ylim([0, 1.2])
        wam_mean_axs.axhline(y=wam_r, color=wamcol, linestyle='dashed', linewidth=meanwidth)

        # bvs_axs[i].axvline(x=c1_mean, color=col1, linestyle='dotted', linewidth=meanwidth)
        # bvs_axs[i].axvline(x=c2_mean, color=col2, linestyle='dashed', linewidth=meanwidth)
        # nvs_axs[i].axvline(x=c1_mean, color=col1, linestyle='dotted', linewidth=meanwidth)
        # nvs_axs[i].axvline(x=c2_mean, color=col2, linestyle='dashed', linewidth=meanwidth)
        # wvs_axs[i].axvline(x=c1_mean, color=col1, linestyle='dotted', linewidth=meanwidth)
        # wvs_axs[i].axvline(x=c2_mean, color=col2, linestyle='dashed', linewidth=meanwidth)
        # wta_axs[i].axvline(x=c1_mean, color=col1, linestyle='dotted', linewidth=meanwidth)
        # wta_axs[i].axvline(x=c2_mean, color=col2, linestyle='dashed', linewidth=meanwidth)
        # wam_axs[i].axvline(x=c1_mean, color=col1, linestyle='dotted', linewidth=meanwidth)
        # wam_axs[i].axvline(x=c2_mean, color=col2, linestyle='dashed', linewidth=meanwidth)

        # RHS tick management
        if i == 2:
            bvs_mean_axs.set_yticks([0,1])
            nvs_mean_axs.set_yticks([0,1])
            wvs_mean_axs.set_yticks([0,1])
            wta_mean_axs.set_yticks([0,1])
            wam_mean_axs.set_yticks([0,1])
        else:
            bvs_mean_axs.set_yticks([])
            nvs_mean_axs.set_yticks([])
            wvs_mean_axs.set_yticks([])
            wta_mean_axs.set_yticks([])
            wam_mean_axs.set_yticks([])

        # Adding labels and inset figures
        if i == 0:
            x = 0.03
            y = 0.7
            input_axs[i].text(x, y, "Cue sample\ndistributions", transform=input_axs[i].transAxes, fontsize="small")

            x0_offset = 0.005#-0.2 #0.005
            x1_offset = 0.072#0.1 #0.07
            y0_offset = 0.01#0 #0.01
            y1_offset = 0.072 # 0.1 #0.07

            xlims = [-0.45, 0.45]
            rect = bvs_axs[i].get_position()
            bvs_inset = fig.add_axes([rect.x0+x0_offset, rect.y0+y0_offset, x1_offset, y1_offset])
            bvs_inset.set_xticks([])
            bvs_inset.set_yticks([])
            bvs_inset.set_xlim(xlims)
            bvs_inset.set_ylim(xlims)
            bvs_inset.set_title("BVS")

            rect = nvs_axs[i].get_position()
            nvs_inset = fig.add_axes([rect.x0+x0_offset, rect.y0+y0_offset, x1_offset, y1_offset])
            nvs_inset.set_xticks([])
            nvs_inset.set_yticks([])
            nvs_inset.set_xlim(xlims)
            nvs_inset.set_ylim(xlims)
            nvs_inset.set_title("NVS")

            rect = wvs_axs[i].get_position()
            wvs_inset = fig.add_axes([rect.x0+x0_offset, rect.y0+y0_offset, x1_offset, y1_offset])
            wvs_inset.set_xticks([])
            wvs_inset.set_yticks([])
            wvs_inset.set_xlim(xlims)
            wvs_inset.set_ylim(xlims)
            wvs_inset.set_title("WVS")

            rect = wta_axs[i].get_position()
            wta_inset = fig.add_axes([rect.x0+x0_offset, rect.y0+y0_offset, x1_offset, y1_offset])
            wta_inset.set_xticks([])
            wta_inset.set_yticks([])
            wta_inset.set_xlim(xlims)
            wta_inset.set_ylim(xlims)
            wta_inset.set_title("WTA")

            rect = wam_axs[i].get_position()
            wam_inset = fig.add_axes([rect.x0+x0_offset, rect.y0+y0_offset, x1_offset, y1_offset])
            wam_inset.set_xticks([])
            wam_inset.set_yticks([])
            wam_inset.set_xlim(xlims)
            wam_inset.set_ylim(xlims)
            wam_inset.set_title("WAM")

            mag = 0.3
            w1 = k1 / (k1+k2)
            w2 = k2 / (k1+k2)
            c1v = cartesian([w1, np.radians(90)])
            c2v = cartesian([w2, np.radians(90 - 120)])
            bvs_out = biased_vector_sum(polar(c1v), polar(c2v))
            bvs_out = cartesian([mag, bvs_out[1]])

            bvs_samples = []
            for idx in range(10):
                bvs_out = biased_vector_sum(polar(c1v), polar(c2v))
                bvs_samples.append(cartesian([mag, bvs_out[1]]))

            origins = np.zeros(3)
            x = -0.25
            y = -0.1
            text_x = 0.9
            text_y = 1.15
            bvs_inset.arrow(x, y, c1v[0], c1v[1], color=col1)
            bvs_inset.arrow(x, y, c2v[0], c2v[1], color=col2)
            for s in bvs_samples:
                bvs_inset.arrow(x, y, s[0], s[1], color=bvscol, width=0.01, alpha=0.5)
#            bvs_inset.text(text_x, text_y, "$g(w \pm N(0, \sigma_{Bias})), +$", transform=bvs_inset.transAxes)


            c1v = cartesian([w1, np.radians(90)])
            c2v = cartesian([w2, np.radians(90 - 120)])
            nvs_out = nonoptimal_vector_sum(polar(c1v), polar(c2v))
            nvs_out = cartesian([mag, nvs_out[1]])
            origins = np.zeros(3)
            nvs_inset.arrow(x, y, c1v[0], c1v[1], color=col1)
            nvs_inset.arrow(x, y, c2v[0], c2v[1], color=col2)
            nvs_inset.arrow(x, y, nvs_out[0], nvs_out[1], color=nvscol, width=0.01)
#            nvs_inset.text(text_x, text_y, "$g(w), +$", transform=nvs_inset.transAxes)

            c1v = cartesian([w1, np.radians(90)])
            c2v = cartesian([w2, np.radians(90 - 120)])
            wvs_out = polar_vector_sum(polar(c1v), polar(c2v))
            wvs_out = cartesian([mag, wvs_out[1]])
            origins = np.zeros(3)
            wvs_inset.arrow(x, y, c1v[0], c1v[1], color=col1)
            wvs_inset.arrow(x, y, c2v[0], c2v[1], color=col2)
            wvs_inset.arrow(x, y, wvs_out[0], wvs_out[1], color=wvscol, width=0.01)

            wta_out = winner_take_all(polar(c1v), polar(c2v))
            wta_out = cartesian([mag, wta_out[1]])
            origins = np.zeros(3)
            # wta_inset.arrow(x, y, c1v[0], c1v[1], color=col1)
            # wta_inset.arrow(x, y, c2v[0], c2v[1], color=col2)
            # wta_inset.arrow(x, y, wta_out[0], wta_out[1], color=wtacol, width=0.01)
            wta_inset.text(0.52, 0.7, r"$w_{B} > w_{G}}$", transform=wta_inset.transAxes, fontsize="small", horizontalalignment='center', verticalalignment='center')
            wta_inset.text(0.5, 0.45, r"$\rightarrow$",
                           transform=wta_inset.transAxes,
                           fontsize="small",
                           horizontalalignment='center',
                           verticalalignment='center')
            wta_inset.text(0.5, 0.2, r"$\theta_{B}$",
                           transform=wta_inset.transAxes,
                           fontsize="small",
                           horizontalalignment='center',
                           verticalalignment='center')

            wam_out = weighted_arithmetic_mean(polar(c1v), polar(c2v))
            wam_out = cartesian([mag, wam_out[1]]) # override returned magnitude
            wam_out = polar(wam_out)
            # wam_inset.arrow(x, y, c1v[0], c1v[1], color=col1)
            # wam_inset.arrow(x, y, c2v[0], c2v[1], color=col2)
            # wam_inset.arrow(x, y, wam_out[0], wam_out[1], color=wamcol, width=0.01)

            wam_inset.text(0.52, 0.5, r"$\frac{w_{B} \theta_{B} + w_{G}\theta_{G}}{2}$", transform=wam_inset.transAxes, fontsize="small", horizontalalignment='center', verticalalignment='center')
            # wam_inset.text(0.5, 0.45, "=",
            #                transform=wam_inset.transAxes,
            #                fontsize="small",
            #                horizontalalignment='center',
            #                verticalalignment='center')

            # wam_inset.text(0.5, 0.2, "{}$^\\circ$".format(int(np.degrees(wam_out[1]))),
            #                transform=wam_inset.transAxes,
            #                fontsize="small",
            #                horizontalalignment='center',
            #                verticalalignment='center')
            # End scope pain
            inset_axs = [bvs_inset, nvs_inset, wvs_inset, wta_inset, wam_inset]

    plt.suptitle("Example output populations under different models", y=0.92)
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    pad = 0.06
    fig.text(0.5, 0.04, 'Simulated exit angle (5$^\circ$ bins)', ha='center')
    fig.text(-0.014+pad, 0.5, '# Individuals', va='center', rotation='vertical')
    fig.text(1-pad, 0.5, 'Mean vector length', va='center', rotation=270)

    fig.savefig("illustrative_model_comparison.svg", bbox_inches='tight')

