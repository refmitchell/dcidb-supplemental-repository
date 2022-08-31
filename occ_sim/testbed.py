"""
testbed.py

Used for as a sandbox for testing various functions and concepts used throughout
the development process. Also used for plotting the outputs of various techniques
used (for example, kappa approximation vs. kappa estimation).
"""
from scipy.special import iv
from scipy.optimize import root_scalar
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import StrMethodFormatter
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import util.models

def adjust_R(R, slope=9):
    """
    R adjustment function; we want to inflate "large" values
    of R, and suppress smaller values
    """
    bias = slope/2
    return 1 / (1 + np.exp(-slope * R + bias))

#
# Finding kappa
#
def f(R, k):
    """
    Function for finding kappa via root finding.

    :param R: Mean vector length
    :param k: kappa
    """
    if np.isnan(iv(0, k)).any():
        print("NaNs found")
    if (iv(0, k) == 0).any():
        print("0vals found")
    if np.isinf(iv(0, k)).any():
        print("Infs found")

    return (iv(1, k) / iv(0, k)) - R

def fn(k, R):
    return f(R, k)

def kappa_approximation(R):
    """
    Kappa MLE approximation from (Mardia and Jupp, 2000, pg. 85,86).
    """
    # For "small" R (5.3.7); R < 0.53
    if R < 0.53:
        return 2*R + R**3 + (5/6)*(R**5)

    # For "large" R (5.3.8); R >= 0.85
    if R >= 0.85:
        return (1 / (2 * (1 - R) - ((1-R)**2) - ((1-R)**3)))
#        return (1 / (2*(1-R))) # (5.3.9) - this isn't a good approximation

    # For "medium" R (5.3.10); 0.53 <= R < 0.85
    return -0.4 + 1.39*R + (0.43/(1-R))

# Finding critical R
def p(n, r):
    R = n*r
    return np.exp(
        np.sqrt(1 + 4*n + 4*(n**2 - R**2)) - (1 + 2*n)
    )

def k(p, n):
    """
    Standard method from Durand and Greenwood 1955
    :param p: significance
    :param n: sample size
    """
    return -np.log(p) - ( (2*np.log(p) + np.log(p)**2) / (4*n) )

def foster_k(p, n):
    return np.sqrt(-np.log(p) / n)

def equivalency():
    """
    Testing equivalency of different example methods for significance.
    """
    p = 0.05
    ret = ( 2*np.log(p) + (np.log(p))**2 ) / 4
    print("ret: " + str(ret))
    print("sqt: " + str(np.sqrt(ret)))
    return foster_k(0.05,10) - np.sqrt(ret)

def rayleigh_Z(n, r):
    """
    Compute rayligh Z statistic

    :param n: sample size
    :param r: r-value (mean vector length)
    """
    return n * (r**2)

def rayleigh_crit(p, n, r):
    """
    Given significance, sample size, and r-value; determine
    if r-val is statistically significant.

    :param p: desired significance
    :param n: sample size
    :param r: r-value (mean vector length)
    """
    Z = rayleigh_Z(n, r)
    K = k(p,n)
    print("p: " + str(p))
    print("n: " + str(n))
    print("r: " + str(r))
    print("Z: " + str(Z))
    print("K: " + str(K))
    if Z > K:
        print("Significant")
    else:
        print("Insignificant")

def vonmises(x, mu, kappa):
    """
    Simple von Mises pdf.

    :param x: target probability density
    :param mu: distribution mean
    :param kappa: distribution concentration
    """
    return np.exp(kappa * np.cos(x - mu)) / (2*np.pi*iv(0, kappa))

def adjusted_weight_function(k0, k1):
    """
    Compute normalised weights and pass them through the sigmoid adjustment.

    :param k0: Kappa 0, the primary kappa for this weight computation
    :param k1: Kappa 1, secondry kappa (alternate cue)
    """
    return adjust_R((k0 / (k1 + k0)), slope=53)

def non_adjusted_weight(k0, k1):
    """
    Normalised un-adjusted weight based on kappas.

    :param k0: Kappa 0, the primary kappa for this weight computation
    :param k1: Kappa 1, secondry kappa (alternate cue)
    """
    return k0 / (k1 + k0)

def mmcs(a1, a2, w1, w2):
    """
    Murray and Morgenstern polar weighted sum

    :param a1: angle one
    :param a2: angle two
    :param w1: weight one
    :param w2: weight two
    """
    # Eq. 16 from Murray and Morgenstern, 2010
    l = a2 + np.arctan2(
        np.sin(a1 - a2),
        (w2 / w1) + np.cos(a1 - a2))
    k = np.sqrt(w2**2 + w1**2 + 2*w2*w1*np.cos(a1 - a2))
    return l, k

def wavg(a1, a2, w1, w2):
    """
    Circular weighted average

    :param a1: angle one
    :param a2: angle two
    :param w1: weight one
    :param w2: weight two
    """
    light = (w1, a1)
    wind = (w2, a2)
    cues = [light, wind]
    cart_cues = [(r*np.cos(theta), r*np.sin(theta)) for (r, theta) in cues]

    xs = [x for (x,_) in cart_cues]
    ys = [y for (_,y) in cart_cues]

    x_avg = sum(xs)/len(xs)
    y_avg = sum(ys)/len(ys)
    r_avg = np.sqrt(x_avg**2 + y_avg**2)
    t_avg = np.arctan2(y_avg, x_avg)

    return t_avg, r_avg

def lwavg(a1, a2, w1, w2):
    """
    A linear weighted average for comparison
    """
    return (w1*a1 + w2*a2), 0


if __name__ == "__main__":
    """
    Most routines here are for plotting; they can be uncommented to produce the
    desired plot.
    """

    """
    Plotting a circular weighted summation vs an weighted arithmetic mean.
    """
    n = 10000
    weights = np.arange(0.1,1,0.1)

    cue_one = np.zeros(n)
    cue_two = np.linspace(0,np.pi,n)

    fig = plt.figure(figsize=(8,4.5))
    plt.subplot(122)

    for weight in weights:
        w1 = weight
        w2 = 1 - weight
        mmcs_out, _ = mmcs(cue_one, cue_two, w1, w2)
        plt.plot(np.degrees(cue_two),
                 np.degrees(mmcs_out),
                 label="w1={:.01f}".format(w1))

    plt.vlines([60,120], 0, 180, color='grey', alpha=0.5, linestyles='--')
    plt.xlim([0,180])
    plt.xticks([0,60,120,180], labels=[r"$0^\circ$",
                                       r"$60^\circ$",
                                       r"$120^\circ$",
                                       r"$180^\circ$",
    ])
    plt.yticks([0,60,120,180], labels=[r"$0^\circ$",
                                       r"$60^\circ$",
                                       r"$120^\circ$",
                                       r"$180^\circ$",
    ])

    plt.ylim([0,180])
    plt.xlabel("Angular position of C2")
    plt.ylabel("WVS Decision Variable L")
    plt.title("Weighted Vector Sum (WVS)")
    plt.gca().set_aspect('equal')

    plt.subplot(121)

    for weight in weights:
        w1 = weight
        w2 = 1 - weight
        wavg_out, _ = lwavg(cue_one, cue_two, w1, w2)
        plt.plot(np.degrees(cue_two),
                 np.degrees(wavg_out),
                 label="w1={:.01f}".format(w1))

    plt.vlines([60,120], 0, 180, color='grey', alpha=0.5, linestyles='--')
    plt.xlim([0,180])
    plt.ylim([0,180])
    plt.xticks([0,60,120,180], labels=[r"$0^\circ$",
                                       r"$60^\circ$",
                                       r"$120^\circ$",
                                       r"$180^\circ$",
    ])
    plt.yticks([0,60,120,180], labels=[r"$0^\circ$",
                                       r"$60^\circ$",
                                       r"$120^\circ$",
                                       r"$180^\circ$",
    ])
    plt.xlabel("Position of C2")
    plt.ylabel("Weighted arithmetic mean")
    plt.title("Weighted Arithmetic Mean (WAM)")
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.legend(fontsize=9)
    plt.savefig("cmle_vs_wam.svg", bbox_inches="tight")
    # plt.show()

    """
    Plotting the outputs of both wavg and mmcs for all possible conflicts.

    Hypothetically, Cue 1 stays at zero, Cue 2 is varied over 0-2pi. We then
    plot the output (Angle between 0-2pi) for all conditions. The point is to
    see how similar these methods are.
    """
    # n = 10000
    # weights = np.arange(0.1,1,0.1)

    # cue_one = np.zeros(n)
    # cue_two = np.linspace(0,np.pi,n)
    # a, l = wavg(0, np.pi, 0.5, 0.5)
    # print("Theta: {}, Length: {}".format(a, l))

    # fig = plt.figure(figsize=(8,6))
    # plt.subplot(221)

    # for weight in weights:
    #     w1 = weight
    #     w2 = 1 - weight
    #     mmcs_out, _ = mmcs(cue_one, cue_two, w1, w2)
    #     plt.plot(cue_two, mmcs_out, label="w1={:.01f}".format(w1))

    # plt.xlim([0,np.pi])
    # plt.ylim([0,np.pi])
    # plt.xlabel("Angular position of C2")
    # plt.ylabel("MMCS Decision Variable L")
    # plt.title("MMCS Output for varying conflict")
    # plt.legend()

    # plt.subplot(222)

    # for weight in weights:
    #     w1 = weight
    #     w2 = 1 - weight
    #     wavg_out, _ = wavg(cue_one, cue_two, w1, w2)
    #     plt.plot(cue_two, wavg_out, label="w1={:.01f}".format(w1))

    # plt.xlim([0,np.pi])
    # plt.ylim([0,np.pi])
    # plt.xlabel("Angular position of C2")
    # plt.ylabel("Weighted average of the cue vectors")
    # plt.title("WAVG output for varying conflict")
    # plt.legend()

    # plt.subplot(223)
    # for weight in weights:
    #     w1 = weight
    #     w2 = 1 - weight
    #     _, mmcs_out = mmcs(cue_one, cue_two, w1, w2)
    #     print(mmcs_out)
    #     plt.plot(cue_two, mmcs_out, label="w1={:.01f}".format(w1))

    # plt.xlim([0,np.pi])
    # plt.ylim([0,1])
    # plt.xlabel("Angular position of C2")
    # plt.ylabel("Length of integrated vector")
    # plt.legend()

    # plt.subplot(224)
    # for weight in weights:
    #     w1 = weight
    #     w2 = 1 - weight
    #     _, wavg_out = wavg(cue_one, cue_two, w1, w2)
    #     plt.plot(cue_two, wavg_out, label="w1={:.01f}".format(w1))

    # plt.xlim([0,np.pi])
    # plt.ylim([0,1])
    # plt.xlabel("Angular position of C2")
    # plt.ylabel("Length of integrated vector")
    # plt.legend()
    # plt.show()

    """
    Plotting normal distributions for individual bias.
    """
    # a, b = -0.05, 0.05

    # bias = 1

    # mu = 0
    # var = 0.00015
    # sigma = np.sqrt(var)
    # xs = np.linspace(a,
    #                  b,
    #                  1000)

    # randoms = np.random.uniform(0, bias, 1000) #np.random.rand(1000)
    # gaussian_randoms = stats.norm.ppf(randoms, mu, sigma)

    # plt.subplot(221)
    # plt.plot(xs, stats.norm.ppf(xs, mu, sigma),
    #          label="Gaussian with $\mu$={} and $\sigma^2$={}".format(mu, var))

    # plt.subplot(222)
    # plt.hist(gaussian_randoms)

    # plt.subplot(223)
    # plt.plot(xs, stats.norm.pdf(xs, mu, sigma),
    #          label="Gaussian with $\mu$={} and $\sigma^2$={}".format(mu, var))
    # plt.xlim([-0.05, 0.05])
    # #plt.ylim([0,1])

    # plt.subplot(224)
    # more_gaussian_randoms = stats.norm.rvs(mu, sigma, 1000)
    # plt.hist(more_gaussian_randoms)
    # plt.legend()

    # plt.show()

    # Plots
    """
    Weight colourmap
    """
    interval=0.01
    kappas = np.arange(0.1, 4+interval, interval)

    k0, k1 = np.meshgrid(kappas, kappas)
    ticks=np.arange(0.5, 4.5, 0.5)
    w = non_adjusted_weight(k1, k0)
    wa = adjusted_weight_function(k1, k0)
    #wa = adjusted_weight_function(kappas, kappas)

    fig = plt.figure(figsize=(8,4.5))
    ax = plt.subplot(121)
    ax.set_aspect('equal')
    ax.set_title("Normalised weight $W_W$ w.r.t. $\kappa_L$, $\kappa_W$", pad=10)
    ax.set_ylabel("$\kappa_W$",fontsize=12)
    ax.set_xlabel("$\kappa_L$",fontsize=12)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.xlim([0.1,4])
    plt.ylim([0.1,4])
    wmap = ax.pcolormesh(k0, k1, w, rasterized=True, shading='auto')
    wmap.set_edgecolor('face')

    ax2 = plt.subplot(122)
    ax2.set_aspect('equal')
    ax2.set_title("Final adjusted weight $w_W = g(W_W)$", pad=10)
    ax2.set_ylabel("$\kappa_W$",fontsize=12)
    ax2.set_xlabel("$\kappa_L$",fontsize=12)
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    plt.xlim([0.1,4])
    plt.ylim([0.1,4])

    wamap = ax2.pcolormesh(k0, k1, wa, rasterized=True, shading='auto')
    wamap.set_edgecolor('face')

    cbar = fig.colorbar(wmap, location='bottom', ax=[ax, ax2], shrink=0.4,orientation='horizontal')# ax=[ax, ax2], shrink=0.6)
    cbar.ax.set_title("Weight value")

    plt.savefig("weight_adjustment_effect.svg", dpi=300, bbox_inches="tight")
    # plt.show()

    """
    Adjusted weight interaction w.r.t. kappas
    """
    # fig = plt.figure(figsize=(12,12))
    # ax = fig.gca(projection='3d')
    # kappas = np.arange(0.1, 4, 0.01)
    # k0, k1 = np.meshgrid(kappas, kappas)
    # w0 = adjusted_weight_function(k0, k1)
    # w1 = adjusted_weight_function(k1, k0)
    # surf = ax.plot_surface(k0, k1, w0, cmap=cm.viridis,
    #                        linewidth=0, antialiased=True,
    #                        alpha=1.0)
    # surf2 = ax.plot_surface(k0, k1, w1, cmap=cm.viridis,
    #                         linewidth=0, antialiased=True,
    #                         alpha=0.2)

    # ax.view_init(elev=28, azim=65)
    # ax.set_title("Relationship between $k$s and resulting adjusted weights")
    # ax.set_xlabel("$\kappa_W$")
    # ax.set_ylabel("$\kappa_L$")

    # cbar = plt.colorbar(surf, shrink=0.5, aspect=5)
    # cbar.ax.set_title("Adjusted weight")
    # plt.savefig("adjusted_weight_function.png", bbox_inches="tight")
    # plt.show()

    """
    Weight function (g)
    """
    # inputs = np.linspace(0, 1, 1000)
    # outputs = adjust_R(inputs, slope=53)
    # ticks = np.arange(0,1.1,step=0.1)
    # print(ticks)
    # fig = plt.figure(figsize=(12,12))
    # plt.plot(inputs, outputs)
    # plt.ylabel("$g(x)$")
    # plt.xlabel("$x$")
    # plt.ylim([0,1])
    # plt.xlim([0,1])
    # plt.xticks(ticks)
    # plt.yticks(ticks)
    # plt.grid(True)
    # plt.title("Weight adjustment function $g(x)$")
    # plt.savefig("weightadj.png", bbox_inches='tight')
    # plt.show()

    # """
    # R value adjustment function
    # """
    # inputs = np.linspace(0, 1, 1000)
    # outputs = adjust_R(inputs)
    # ticks = np.arange(0,1.1,step=0.1)
    # print(ticks)
    # fig = plt.figure(figsize=(12,12))
    # plt.plot(inputs, outputs, label="$h(R)$")
    # plt.ylabel("$R'$")
    # plt.xlabel("$R$ value (mean vector length)")
    # plt.ylim([0,1])
    # plt.xlim([0,1])
    # plt.xticks(ticks)
    # plt.yticks(ticks)
    # plt.grid(True)
    # plt.title("$R$ value adjustment for $\kappa$ estimation")
    # plt.legend()
    # plt.savefig("rvaladj.png", bbox_inches='tight')
    # plt.show()

    """
    von Mises distributions for different Kappa
    """
    # thetas = np.linspace(-np.pi, np.pi, 1000)
    # kappas = [0.25, 0.5, 1, 2, 4, 8]

    # fig = plt.figure(figsize=(4,3))
    # for kappa in kappas:
    #     yvals = vonmises(thetas, 0, kappa)
    #     plt.plot(np.degrees(thetas), yvals, label="$\kappa =$ {}".format(kappa))

    # plt.ylabel("Probability density")
    # plt.xlabel("x (Degrees)")
    # plt.ylim([0,1])
    # plt.xlim([-180, 180])
    # plt.title("von Mises probability density function")
    # plt.legend()
    # plt.savefig("vmpdf.png", bbox_inches='tight')
    # plt.show()



    # """
    # Kappa approximation vs. true solutions
    # """
    kvals = np.linspace(0, 300, 1000)
    rvals = np.linspace(0, 0.98, 1000)
    kr = [ kappa_approximation(x) for x in rvals]
    R = 0.998

    fun = f(R, kvals)
    roots = []
    for r in rvals:
        res = root_scalar(fn, args=(r), bracket=[0, 500], x0=10, x1=700)
        roots.append(res.root)

    fig = plt.figure(figsize=(8,4.5))
    plt.subplot(111)
    plt.plot(rvals, kr, label="$\kappa$ approximation", color="red", linestyle='dashdot')
    plt.plot(rvals, roots, label="True $\kappa$ estimates", color="blue", alpha=0.5)
    plt.title(r'$\hat\kappa$ approximation vs. true solutions of $R =\frac{ I_1 (\kappa ) }{ I_0 (\kappa ) }$')
    plt.ylabel("$\hat\kappa$")
    plt.xlabel("Mean vector length - $R$")
    plt.legend()

    plt.savefig("kappa_approximation.svg", bbox_inches='tight')
    # plt.show()
