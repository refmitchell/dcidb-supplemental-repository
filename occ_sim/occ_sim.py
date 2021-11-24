"""
occ_sim.py

A graphical interface for the basic software. This allows you to try out the
different simulators with different parameters to see what behaviour one would
expect on a trial-by-trial basis. This is helpful for visualisation, but
does not give a meaningful insight into the fit of each model to the available
data.

To run this, make sure you have the required anaconda environment configured;
then simply use;

$: python occ_sim.py
"""

# Library imports
import sys
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QFile, Slot
from gui.ui_mainwindow import Ui_MainWindow

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from scipy.special import i0
import numpy as np
import time


# Internal imports
from util.integration_models import *
from util.models import ReliabilityModel

from world.light import Light
from world.wind import Wind

from util.treatment import Treatment


# MainWindow wrapper
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Link UI Actions.
        self.ui.refresh_button.clicked.connect(self.run_simulation)
        self.ui.sim_combobox.currentTextChanged.connect(self.update_options)

        # MPL Canvas
        self.canvas = FigureCanvas(Figure(figsize=(16,9)))
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.behaviour_ax = self.canvas.figure.add_subplot(1,2,1,
                                                           projection='polar')

        self.init_dist_ax = self.canvas.figure.add_subplot(2,2,2)
        self.conf_dist_ax = self.canvas.figure.add_subplot(2,2,4)

        # Blank display
        self.behaviour_ax.set_rlim(0,1.1)
        self.behaviour_ax.set_theta_zero_location("N")
        self.behaviour_ax.set_theta_direction(-1)

        self.ui.mpl_layout_space.addWidget(self.toolbar)
        self.ui.mpl_layout_space.addWidget(self.canvas)

    #
    # UI Callbacks
    #
    @Slot()
    def update_options(self):
        """
        Update options panel in response to simulator change; different
        sims allow different options.
        """
        # 0 = blank (should always be inactive)
        # 1 = BWS
        # 2 = NWS
        # 3 = SingleCue
        if self.ui.sim_combobox.currentText() == "Non-optimal Weighted Sum":
            self.ui.options_stack.setEnabled(True)
            self.ui.options_stack.setCurrentIndex(2)
        elif self.ui.sim_combobox.currentText() == "Single Cue":
            self.ui.options_stack.setEnabled(True)
            self.ui.options_stack.setCurrentIndex(3)
        elif self.ui.sim_combobox.currentText() == "Biased non-optimal Weighted Sum":
            self.ui.options_stack.setEnabled(True)
            self.ui.options_stack.setCurrentIndex(1)
        else:
            self.ui.options_stack.setEnabled(False)
            self.ui.options_stack.setCurrentIndex(0)

    @Slot()
    def run_simulation(self):
        """
        Run a full simulation with selected options
        """
        # Choose simulator
        simulator_text = self.ui.sim_combobox.currentText()
        simulator = self.__get_sim_from_str(simulator_text)

        #
        # Construct treatment
        #
        rel_model = ReliabilityModel()
        treatment = Treatment()
        treatment.set_reliability_model(rel_model)
        treatment.set_n(self.ui.n_spinbox.value())
        treatment.set_id(self.ui.title_lineedit.text())

        init_light = Light(
            np.radians(self.ui.init_light_elevation_spinbox.value()),
            np.radians(self.ui.init_light_azimuth_spinbox.value()),
            treatment
            )

        init_wind = Wind(
            self.ui.init_wind_speed_spinbox.value(),
            np.radians(self.ui.init_wind_azimuth_spinbox.value()),
            treatment
            )

        initial = [init_wind, init_light]

        conf_light = Light(
            np.radians(self.ui.conf_light_elevation_spinbox.value()),
            np.radians(self.ui.conf_light_azimuth_spinbox.value()),
            treatment
            )

        conf_wind = Wind(
            self.ui.conf_wind_speed_spinbox.value(),
            np.radians(self.ui.conf_wind_azimuth_spinbox.value()),
            treatment
            )

        conflict = [conf_wind, conf_light]

        treatment.set_initial_cues(initial)
        treatment.set_conflict_cues(conflict)

        # Timing
        start_time = time.time()

        # Run simulation
        simulator.simulate_treatment(treatment)

        # Timing included to give a sense of how long each method took to
        # run. This does change depending on the machine or activity.
        print("Runtime = {}".format(time.time() - start_time))

        #
        # Plotting
        #

        # Clear existing axes
        self.behaviour_ax.cla()
        self.init_dist_ax.cla()
        self.conf_dist_ax.cla()

        changes = treatment.get_changes_in_bearing()
        avg_r, avg_t = treatment.get_avg_change()
        print("R: {}".format(avg_r))
        print("Mu: {}".format(np.degrees(avg_t)))

        # Bin data into 360/nbins degree bins to plot the population mass
        nbins = 72
        ch_hist = np.histogram(np.degrees(changes), np.linspace(-180, 180, nbins + 1))[0]
        ch_hist_norm = ch_hist / sum(ch_hist)

        self.behaviour_ax.plot(changes, np.ones(len(changes)), 'bo', color='magenta', alpha=0.2)
        self.behaviour_ax.plot(avg_t, avg_r, 'ro', markeredgecolor='k', label="R={:.02f},Th={:.01f}".format(avg_r, np.degrees(avg_t)))
        self.behaviour_ax.set_title("{}: {}".format(simulator.name(), treatment.get_id()))
        self.behaviour_ax.set_rlim(0,1.1)
        self.behaviour_ax.set_theta_zero_location("N")
        self.behaviour_ax.set_theta_direction(-1)
        self.behaviour_ax.legend(loc='lower left')

        params = treatment.get_cue_distribution_parameters()

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
        self.init_dist_ax.plot(np.degrees(light_x), light_y, color='green', label="Light: kappa={:.2f}".format(light_kappa))
        self.init_dist_ax.plot(np.degrees(wind_x), wind_y, color='blue', label="Wind: kappa={:.2f}".format(wind_kappa))
        self.init_dist_ax.set_ylim([0,1])
        self.init_dist_ax.legend()
        self.init_dist_ax.set_title("Initial cue probability distributions")
        self.init_dist_ax.set_xlabel("Degrees")
        self.init_dist_ax.set_ylabel("Probability density")


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
        self.conf_dist_ax.bar(np.linspace(-180, 180, nbins), ch_hist_norm, width=360/nbins, color='magenta',edgecolor='k', label='Population response', alpha=0.5)

        self.conf_dist_ax.plot(np.degrees(light_x), light_y, color='green', label="Light: kappa={:.2f}".format(light_kappa))
        self.conf_dist_ax.plot(np.degrees(wind_x), wind_y, color='blue', label="Wind: kappa={:.2f}".format(wind_kappa))
        self.conf_dist_ax.set_ylim([0,1])
        self.conf_dist_ax.set_xlim([-180,180])
        self.conf_dist_ax.legend()

        self.conf_dist_ax.set_title("Conflict cue probability distributions")
        self.conf_dist_ax.set_xlabel("Degrees")
        self.conf_dist_ax.set_ylabel("Probability density")

        # Redraw the figure
        self.canvas.draw()

    def __get_sim_from_str(self, name):
        """
        Decode the text from the simulator combobox and return the
        correct simulator (with options assumed to have been set).

        :param name: The textual name in the drop-down selector at the time
                     the refresh button is clicked.
        :return: The correct simulator instance.
        """
        if name == "Weighted Vector Sum":
            return CMLE()
        elif name == "Non-optimal Weighted Sum":
            return BWS(
                adjustment_slope=self.ui.nws_slope_spinbox.value(),
                bias_variance=0
            )
        elif name == "Biased non-optimal Weighted Sum":
            return BWS(
                adjustment_slope=self.ui.bws_slope_spinbox.value(),
                bias_variance=self.ui.bws_variance_spinbox.value(),
                no_window=True
            )
        elif name == "Weighted Arithmetic Mean":
            return WAM()
        elif name == "Winner Take All":
            return WTA()
        elif name == "Single Cue":
            return SingleCue(cue=self.ui.sc_cue_combobox.currentText().lower())

        # Catch-all, should never happen
        return Simulator()

if __name__=="__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
