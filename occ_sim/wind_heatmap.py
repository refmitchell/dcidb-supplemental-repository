"""
wind_heatmap.py

This module was used for producing the wind speed maps in the supplemental
information. The data was collected in the lab in Lund using polar positions
from the centre of the experimental arena. It was then translated to
cartesian coordinates for interpolation then translated back to produce
the final polar plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d,griddata,bisplrep,bisplev

if __name__ == "__main__":
    df = pd.read_csv("data/wind_heatmap_data.csv")

    ths = np.radians(np.array(df["theta"]))
    rs = np.array(df["m"])
    tf = np.array(df["two-five"])
    otf = np.array(df["one-two-five"])

    n = 1000
    tt = np.linspace(0,360,n)
    rr = np.linspace(0,30,n)
    tt, rr = np.meshgrid(tt,rr)

    # Convert to cartesian
    xs = np.zeros(len(df))
    ys = np.zeros(len(df))

    for i in range(len(df)):
        xs[i] = rs[i]*np.cos(ths[i])
        ys[i] = rs[i]*np.sin(ths[i])

    # Linspaces for x and y
    xx = np.linspace(min(xs), max(xs), n)
    yy = np.linspace(min(ys), max(ys), n)
    xx, yy = np.meshgrid(xx, yy)

    interp_tf = griddata((xs,ys), tf, (xx.ravel(), yy.ravel()), method='linear')
    interp_tf = interp_tf.reshape(n,n)

    interp_otf = griddata((xs,ys), otf, (xx.ravel(), yy.ravel()), method='linear')
    interp_otf = interp_otf.reshape(n,n)

    rr = np.zeros((n,n))
    tt = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            rr[i][j] = np.sqrt(xx[i][j]**2 + yy[i][j]**2)
            tt[i][j] = np.arctan2(yy[i][j], xx[i][j])
    #
    # Plotting
    #
    plt.figure(figsize=(8,6.5))
    ax = plt.subplot(121, projection="polar")
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim([0,30])
    ax.set_rticks([0,10,20,30])

    ax.set_title("Wind speed across arena, 2.5m/s")

    cmap = ax.pcolormesh(tt, rr, interp_tf, rasterized=True, cmap='Blues', shading='auto')
    cmap.set_edgecolor('face')
    ax.grid(True, color='k')
    plt.colorbar(cmap, ax=ax, orientation='horizontal', pad=0.05)

    ax2 = plt.subplot(122, projection="polar")
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_rlim([0,30])
    ax2.set_rticks([0,10,20,30])
    ax2.set_title("Wind speed across arena, 1.25m/s")

    cmap2 = ax2.pcolormesh(tt, rr, interp_otf, rasterized=True, cmap='Blues', shading='auto')
    cmap2.set_edgecolor('face')
    ax2.grid(True, color='k')
    plt.colorbar(cmap2, ax=ax2, orientation="horizontal", pad=0.05)
    plt.tight_layout()
    plt.savefig("wind_heatmaps.svg", dpi=300, bbox_inches="tight")
    plt.show()
