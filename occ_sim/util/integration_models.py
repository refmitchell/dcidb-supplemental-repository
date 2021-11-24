"""
integration_models.py

This module contains all of the descriptive models trialled for the actual
integration step. They all take input in the form of a Treatment and then
perform the respective simulation. If a Treatment has pre-generated samples
then the simulators will use these to avoid variance during comparison scenarios
(with the exception of SingleCue).
"""

import numpy as np
from scipy.stats import norm
import sys

def sigmoid(x, slope=30, bias=0.5):
    """
    Sigmoid adjustment with the purpose of inflating values above 0.5 and
    suppressing those below.

    :param x: input value
    :param slope: the slope of the sigmoid (how steep it is)
    :param bias: the horizontal shift (should always be 0.5)
    :return: The adjusted value.
    """
    return 1 / (1 + np.exp(-slope *(x - bias)))

class Simulator():
    """
    Simulator base class; allows some error checking without a complete crash.
    """
    def __init__(self,output=False):
        """
        :param output: turn on or off output for this simulation.
        """
        self.__output=output
        self.__name = "BaseSimulator"
        self.__samples = None # Expects dict

    def check_options(self, treatment):
        """
        [Legacy]

        Check the optional settings from the config to make sure they are
        sensible. This is a lingering check which made sense when using
        config files, less so without them. Retained as configs can technically
        still be used (though this is not recommended).

        :param treatment: the configured Treatment
        :return: boolean indicating option validity
        """
        mode = treatment.get_mode()

        #
        # Originally this would support multiple modes and variable numbers of
        # cues; however, this was added in differently. SingleCue is used for
        # single cue scenarios and change in accuracy simulations were only
        # required for initial cue characterisation (and easy to include without
        # this explicit option).
        #
        if mode == "change-in-accuracy":
            print("Stub, accuracy mode not yet implemented.")
            return False
        elif mode == "change-in-bearing":
            if len(treatment.get_initial_cues()) < 2:
                print("Stub, two cues -> two cues is all that's supported for now")
                return False
        return True

    def output(self):
        """
        self.__output getter
        :return: output setting for this simulator
        """
        return self.__output

    def simulate_treatment(self):
        """
        Catch-all to prevent uninformative crashes.
        """
        sys.exit("Fatal: simulation mechanism not yet implemented")

    def set_name(self, name):
        """
        Simulator name setter

        :param name: the desired simlator name
        """
        self.__name = name

    def name(self):
        """
        Simulator name getter
        :return: the set name of this simulator
        """
        return self.__name

class SingleCue(Simulator):
    """
    Utility class for testing population accuracy under single cues
    """
    def __init__(self, output=False, cue="light"):
        """
        Constructor; set cue to follow and simulator name.

        :param output: enable or disable output
        :param cue: cue to follow
        """
        super().__init__(output)
        self.__cue = cue
        self.set_name("SingleCue")

    def simulate_treatment(self, treatment):
        """
        Core simulation routine
        :param treatment: the treatment to be simulated
        """
        if not self.check_options(treatment):
            return False

        # Initialise
        initial_cue = None
        shifted_cue = None

        # Retrieve the cue information for the target cue
        if self.__cue == "light":
            initial_cue = [x for x in treatment.get_initial_cues()
                           if x.get_type() == "light"][0]
            shifted_cue = [x for x in treatment.get_conflict_cues()
                           if x.get_type() == "light"][0]
        else:
            initial_cue = [x for x in treatment.get_initial_cues()
                           if x.get_type() == "wind"][0]
            shifted_cue = [x for x in treatment.get_conflict_cues()
                           if x.get_type() == "wind"][0]

        # Sample angles from cue distributions
        initial_samples = initial_cue.sample(treatment.get_n())
        shifted_samples = shifted_cue.sample(treatment.get_n())

        changes = []
        for trial in range(treatment.get_n()):
            initial_sample = initial_samples[trial]
            shifted_sample = shifted_samples[trial]

            change = shifted_sample - initial_sample
            changes.append(change)

            if self.output():
                print("Trial #{}".format(treatment.get_n()))
                print("Initial sample angle: " + str(np.degrees(initial_sample)))
                print("Conflict sample angle: " + str(np.degrees(conflict_sample)))

            changes.append(change)

        # Store results in the treatment object
        treatment.set_changes_in_bearing(changes)

class WAM(Simulator):
    """
    Weighted Arithmetic Mean: simple linear weighted average. This is to be
    included as a point of comparison so we can show a difference between
    the linear case and the circular case.
    """
    def __init__(self, output=False):
        """
        Constructor
        :param output: enable or disable output
        """
        super().__init__(output)
        self.set_name("WAM")

    def compute_integration(self,
                            light,
                            light_weight,
                            wind,
                            wind_weight,
                            offset):
        """
        Compute the cue integration

        :param light: the azimuthal angle of the light cue
        :param light_weight: the relative weight of the light cue
        :param wind: the azimuthal angle of the wind cue
        :param wind_weight: the relative weight of the wind cue
        """
        adj_wind = wind - offset

        return light*light_weight + wind*wind_weight

    def simulate_treatment(self, treatment):
        """
        Core simulation routine
        :param treatment: the treatment to be simulated
        """
        if self.check_options(treatment):
            #
            # Extract positional information for the cues
            #
            light = [x for x in treatment.get_initial_cues()
                     if x.get_type() == "light"][0]
            wind = [x for x in treatment.get_initial_cues()
                    if x.get_type() == "wind"][0]

            conflict_light = [x for x in treatment.get_conflict_cues()
                              if x.get_type() == "light"][0]
            conflict_wind = [x for x in treatment.get_conflict_cues()
                             if x.get_type() == "wind"][0]

            # Initialise sample variables
            wind_samples = None
            light_samples = None
            conflict_wind_samples = None
            conflict_light_samples = None

            # If this treatment had preset samples for testing,
            # use those rather than generating fresh ones
            if treatment.preset_samples():
                samples = treatment.get_samples()
                initial = samples["initial"]
                conflict = samples["conflict"]

                wind_samples = initial["wind"]
                light_samples = initial["light"]
                conflict_wind_samples = conflict["wind"]
                conflict_light_samples = conflict["light"]
                print("WAM: Using preset samples")
            else:
                wind_samples = wind.sample(treatment.get_n())
                light_samples = light.sample(treatment.get_n())

                conflict_wind_samples = conflict_wind.sample(treatment.get_n())
                conflict_light_samples = conflict_light.sample(treatment.get_n())

            print("WAM: simulating " + str(treatment.get_n()) +
                  " trials" + " for treatment " + treatment.get_id())

            changes = []
            for trial in range(treatment.get_n()):
                # Take sample for each trial
                light_sample = light_samples[trial]
                wind_sample = wind_samples[trial]
                conflict_light_sample = conflict_light_samples[trial]
                conflict_wind_sample = conflict_wind_samples[trial]

                # Retrieve cue weights - these are the kappa parameters
                light_weight_clean = light.get_weight()
                wind_weight_clean = wind.get_weight()
                conflict_light_weight_clean = conflict_light.get_weight()
                conflict_wind_weight_clean = conflict_wind.get_weight()

                # Normalise weights
                light_weight = light_weight_clean / (light_weight_clean + wind_weight_clean)
                wind_weight = wind_weight_clean / (light_weight_clean + wind_weight_clean)
                conflict_light_weight = conflict_light_weight_clean / (conflict_light_weight_clean + conflict_wind_weight_clean)
                conflict_wind_weight = conflict_wind_weight_clean / (conflict_light_weight_clean + conflict_wind_weight_clean)

                # Angular offset between cues has to be taken into account
                offset = wind_sample - light_sample

                # Initial "roll" integration
                bearing = self.compute_integration(
                    light_sample,
                    light_weight,
                    wind_sample,
                    wind_weight,
                    offset)

                # Conflict "roll" integration
                conflict_bearing = self.compute_integration(
                    conflict_light_sample,
                    conflict_light_weight,
                    conflict_wind_sample,
                    conflict_wind_weight,
                    offset)

                # Change in bearing
                change = conflict_bearing - bearing
                changes.append(change)

            treatment.set_changes_in_bearing(changes)

class CMLE(Simulator):
    """
    Circular Maximum Likelihood Estimate
    [Note, in the paper this was referred to as: Weighted Vector Sum]
    From: Cue integration on the circle and the sphere
    By: Murray and Morgenstern 2010

    Weighted summation with statistically optimal weights.
    """
    def __init__(self, output=False):
        """
        Constructor
        :param output: enable or disable output
        """
        super().__init__(output)
        self.set_name("CMLE")

    def compute_integration(self,
                            light,
                            light_weight,
                            wind,
                            wind_weight,
                            offset):
        """
        Compute the cue integration

        :param light: the azimuthal angle of the light cue
        :param light_weight: the relative weight of the light cue
        :param wind: the azimuthal angle of the wind cue
        :param wind_weight: the relative weight of the wind cue
        """
        # Account for any offset from the initial presentation
        adj_wind = wind - offset

        # Eq. 16 from Murray and Morgenstern, 2010
        l = adj_wind + np.arctan2(
            np.sin(light - adj_wind),
            (wind_weight / light_weight) + np.cos(light - adj_wind))

        return l

    def simulate_treatment(self, treatment):
        """
        Core simulation routine
        :param treatment: the treatment to be simulated
        """
        if self.check_options(treatment):
            # Extract cue information
            light = [x for x in treatment.get_initial_cues()
                     if x.get_type() == "light"][0]
            wind = [x for x in treatment.get_initial_cues()
                    if x.get_type() == "wind"][0]

            conflict_light = [x for x in treatment.get_conflict_cues()
                              if x.get_type() == "light"][0]
            conflict_wind = [x for x in treatment.get_conflict_cues()
                             if x.get_type() == "wind"][0]

            # Initialise sample variables
            wind_samples = None
            light_samples = None
            conflict_wind_samples = None
            conflict_light_samples = None

            # If this treatment had preset samples for testing,
            # use those rather than generating fresh ones
            if treatment.preset_samples():
                samples = treatment.get_samples()
                initial = samples["initial"]
                conflict = samples["conflict"]

                wind_samples = initial["wind"]
                light_samples = initial["light"]
                conflict_wind_samples = conflict["wind"]
                conflict_light_samples = conflict["light"]
                print("CMLE: Using preset samples")
            else:
                wind_samples = wind.sample(treatment.get_n())
                light_samples = light.sample(treatment.get_n())

                conflict_wind_samples = conflict_wind.sample(treatment.get_n())
                conflict_light_samples = conflict_light.sample(treatment.get_n())

            print("CMLE: simulating " + str(treatment.get_n()) +
                  " trials" + " for treatment " + treatment.get_id())

            changes = []
            for trial in range(treatment.get_n()):
                # Get sample for each trial
                light_sample = light_samples[trial]
                wind_sample = wind_samples[trial]
                conflict_light_sample = conflict_light_samples[trial]
                conflict_wind_sample = conflict_wind_samples[trial]

                # Retrieve cue weights - these are the kappa parameters
                light_weight_clean = light.get_weight()
                wind_weight_clean = wind.get_weight()
                conflict_light_weight_clean = conflict_light.get_weight()
                conflict_wind_weight_clean = conflict_wind.get_weight()

                # Normalise weights
                light_weight = light_weight_clean / (light_weight_clean + wind_weight_clean)
                wind_weight = wind_weight_clean / (light_weight_clean + wind_weight_clean)
                conflict_light_weight = conflict_light_weight_clean / (conflict_light_weight_clean + conflict_wind_weight_clean)
                conflict_wind_weight = conflict_wind_weight_clean / (conflict_light_weight_clean + conflict_wind_weight_clean)

                # Angular offset between cues has to be taken into account
                offset = wind_sample - light_sample

                # Initial "roll"
                bearing = self.compute_integration(
                    light_sample,
                    light_weight,
                    wind_sample,
                    wind_weight,
                    offset)

                # Conflict "roll"
                conflict_bearing = self.compute_integration(
                    conflict_light_sample,
                    conflict_light_weight,
                    conflict_wind_sample,
                    conflict_wind_weight,
                    offset)

                # Change in bearing
                change = conflict_bearing - bearing

                if self.output():
                    print("Trial #{}".format(treatment.get_n()))
                    print("light_sample angle: " + str(np.degrees(light_sample)))
                    print("light weight: " + str(conflict_light_weight))
                    print("wind_sample angle: " + str(np.degrees(wind_sample)))
                    print("wind weight: " + str(conflict_wind_weight))
                    print("Initial integration: " + str(np.degrees(bearing)))
                    print("Conflict integration: " + str(np.degrees(conflict_bearing)))
                    print("Change: " + str(np.degrees(change)))

                changes.append(change)

            # Store output in treatment
            treatment.set_changes_in_bearing(changes)

class BWS(Simulator):
    """
    Biased non-optimal Weighted Sum

    Injecting individuals with a little bit of noise before
    adjusting the weights. Each individual will get a slight bias
    towards one cue.

    This is NWS if the bias_window parameter is set to -1 and no_window is set to
    False (default). The concept of bias windows was removed fairly late in
    development, as such they are prevelant in code. For BWS as it appeared in
    the paper, no_window should be True (i.e. ignore bias windows).

    Default parameters are fit as maximally likely w.r.t. the experimental
    data.
    """
    def __init__(self,
                 output=False,
                 adjustment_slope=53,
                 adjustment_bias=0.5,
                 bias_variance=0.0003,
                 bias_window=0.02,
                 no_window=False
    ):
        """
        Constructor

        :param output: enable/disavle output for this simulator
        :param adjustment_slope: set the adjustment function slope parameter
        :param adjustment_bias: set the adjustment function bias parameter
        :param bias_variance: set the variance of the bias distribution (Gaussian)
        :param bias_window: define a region in weight-space where biases
                            will have an effect (-1 == NWS).
        :param no_window: if True, biases are used but bias windows are not
        """
        super().__init__(output)
        self.__bias_mu = 0
        self.__bias_variance = bias_variance
        self.__bias_sigma = np.sqrt(self.__bias_variance)
        self.__adjustment_slope = adjustment_slope
        self.__adjustment_bias = adjustment_bias
        self.__bias_window = bias_window
        self.__no_window = no_window


        # Bias window of -1 indicates no biases are to be applied
        if self.__bias_window == -1:
            self.set_name("NWS-{}-{}".format(adjustment_slope, adjustment_bias))
        else:
            # Where this name is important, adjustment parameters should be fixed
            self.set_name("BWS-{}-{}-{}".format(bias_window,
                                                adjustment_slope,
                                                bias_variance))

    def compute_integration(self,
                            light,
                            light_weight,
                            wind,
                            wind_weight,
                            offset):
        """
        Compute the cue integration

        :param light: the azimuthal angle of the light cue
        :param light_weight: the relative weight of the light cue
        :param wind: the azimuthal angle of the wind cue
        :param wind_weight: the relative weight of the wind cue
        """
        # Account for any offset from the initial presentation
        adj_wind = wind - offset

        # Eq. 16 from Murray and Morgenstern, 2010
        l = adj_wind + np.arctan2(
            np.sin(light - adj_wind),
            (wind_weight / light_weight) + np.cos(light - adj_wind))

        return l

    def simulate_treatment(self, treatment):
        """
        Core simulation routine
        :param treatment: the treatment to be simulated
        """
        if self.check_options(treatment):
            # Extract cue position information
            light = [x for x in treatment.get_initial_cues()
                     if x.get_type() == "light"][0]
            wind = [x for x in treatment.get_initial_cues()
                    if x.get_type() == "wind"][0]

            conflict_light = [x for x in treatment.get_conflict_cues()
                              if x.get_type() == "light"][0]
            conflict_wind = [x for x in treatment.get_conflict_cues()
                             if x.get_type() == "wind"][0]

            # Initialiase sample variables
            wind_samples = None
            light_samples = None
            conflict_wind_samples = None
            conflict_light_samples = None

            # If this treatment had preset samples for testing,
            # use those rather than generating fresh ones
            if treatment.preset_samples():
                samples = treatment.get_samples()
                initial = samples["initial"]
                conflict = samples["conflict"]

                wind_samples = initial["wind"]
                light_samples = initial["light"]
                conflict_wind_samples = conflict["wind"]
                conflict_light_samples = conflict["light"]
            else:
                wind_samples = wind.sample(treatment.get_n())
                light_samples = light.sample(treatment.get_n())

                conflict_wind_samples = conflict_wind.sample(treatment.get_n())
                conflict_light_samples = conflict_light.sample(treatment.get_n())

            # Initialise biases to zero (i.e. assume no bias)
            biases = np.zeros(treatment.get_n())

            # Biases are small and normally distributed about 0
            # We expect most beetles to have little to no bias
            # a few end up with an extreme bias. Ignore if bias
            # parameters indicate no biases should be used or if
            # variance is zero (as norm.rvs will fail)

            # Case where we want to ignore bias windows, also need to
            # check variance is non-zero
            ignore_window = self.__no_window and self.__bias_variance != 0

            # Case where windows are employed, check values are sensible
            with_window = self.__bias_window != -1 and self.__bias_variance != 0

            # If configured correctly, draw biases from the specified normal
            # distribution
            if ignore_window or with_window:
                biases = norm.rvs(self.__bias_mu,
                                  self.__bias_sigma,
                                  treatment.get_n())

            if (self.output()):
                print("{}: simulating {} trials for treatment {}"
                      .format(
                          self.name(),
                          self.treatment.get_n(),
                          treatment.get_id()
                      )
                )

            changes = []
            for trial in range(treatment.get_n()):
                # Draw trial samples and trial bias
                light_sample = light_samples[trial]
                wind_sample = wind_samples[trial]
                conflict_light_sample = conflict_light_samples[trial]
                conflict_wind_sample = conflict_wind_samples[trial]
                bias = biases[trial]

                # Retrieve cue weights - these are the kappa parameters
                light_weight_clean = light.get_weight()
                wind_weight_clean = wind.get_weight()
                conflict_light_weight_clean = conflict_light.get_weight()
                conflict_wind_weight_clean = conflict_wind.get_weight()

                # Normalise weights
                light_weight = light_weight_clean / (light_weight_clean + wind_weight_clean)
                wind_weight = wind_weight_clean / (light_weight_clean + wind_weight_clean)

                conflict_light_weight = conflict_light_weight_clean / (conflict_light_weight_clean + conflict_wind_weight_clean)
                conflict_wind_weight = conflict_wind_weight_clean / (conflict_light_weight_clean + conflict_wind_weight_clean)

                # bias adjustment w.r.t. inverse weight difference
                w_diff = abs(light_weight - wind_weight)
                inv_diff = 1 - w_diff


                # Bias has no effect outside the bias window
                # Only when cues are sufficiently closely weighted does
                # bias come into play. Can be thought of as a weak prior.

                # If windows are in use
                if not self.__no_window:
                    # Check weight difference lies within the bias window
                    if w_diff > self.__bias_window:
                        bias = 0 # If not, eliminate bias

                # Include bias
                light_weight = light_weight + bias
                wind_weight = wind_weight - bias
                conflict_light_weight = conflict_light_weight + bias
                conflict_wind_weight = conflict_wind_weight - bias

                # Non-zero adjustment slope, include adjustment.
                # adjustment slope of zero is treated as no adjustment (i.e.
                # normalised weights are used.) This allows comparison of
                # adjusted values to linear ones, we can see the shape of the
                # full parameter continuum.
                if self.__adjustment_slope != 0:
                    light_weight = sigmoid(light_weight,
                                           slope=self.__adjustment_slope,
                                           bias=self.__adjustment_bias
                    )
                    wind_weight = sigmoid(wind_weight,
                                          slope=self.__adjustment_slope,
                                          bias=self.__adjustment_bias)
                    conflict_light_weight = sigmoid(conflict_light_weight,
                                                    slope=self.__adjustment_slope,
                                                    bias=self.__adjustment_bias)
                    conflict_wind_weight = sigmoid(conflict_wind_weight,
                                                   slope=self.__adjustment_slope,
                                                   bias=self.__adjustment_bias
                    )

                # Angular offset between cues has to be taken into account
                offset = wind_sample - light_sample

                # Initial "roll"
                bearing = self.compute_integration(
                    light_sample,
                    light_weight,
                    wind_sample,
                    wind_weight,
                    offset)

                # Conflict "roll"
                conflict_bearing = self.compute_integration(
                    conflict_light_sample,
                    conflict_light_weight,
                    conflict_wind_sample,
                    conflict_wind_weight,
                    offset)

                # Change in bearing
                change = conflict_bearing - bearing

                if self.output():
                    print("Trial #{}".format(treatment.get_n()))
                    print("light_sample angle: " + str(np.degrees(light_sample)))
                    print("light weight: " + str(conflict_light_weight))
                    print("wind_sample angle: " + str(np.degrees(wind_sample)))
                    print("wind weight: " + str(conflict_wind_weight))
                    print("Initial integration: " + str(np.degrees(bearing)))
                    print("Conflict integration: " + str(np.degrees(conflict_bearing)))
                    print("Change: " + str(np.degrees(change)))

                changes.append(change)

            # Store results in treatment object
            treatment.set_changes_in_bearing(changes)

class WTA(Simulator):
    """
    Winner Take All (note that can be parameterised as Biased WTA)

    Cue with the greatest weight wins.
    """
    def __init__(self, output=False, bias_window=0, bias_variance=0):
        """
        Constructor
        :param output: enable or disable output
        """
        super().__init__(output)
        self.set_name("WTA")

    def compute_integration(self,
                            light_sample,
                            light_weight,
                            wind_sample,
                            wind_weight,
                            offset):
        """
        Compute the cue integration

        :param light: the azimuthal angle of the light cue
        :param light_weight: the relative weight of the light cue
        :param wind: the azimuthal angle of the wind cue
        :param wind_weight: the relative weight of the wind cue
        """
        adj_wind = wind_sample - offset
        if light_weight > wind_weight:
            return light_sample
        elif light_weight < wind_weight:
            return adj_wind
        else:
            # This case is vanishingly unlikely and does not occur in the
            # biological data against which we evaluate the models. If one were
            # to see equal weights, one could perform a random choice between the
            # two cues.
            return light_sample

    def simulate_treatment(self, treatment):
        """
        Core simulation routine
        :param treatment: the treatment to be simulated
        """
        if self.check_options(treatment):
            # Get cue position information
            light = [x for x in treatment.get_initial_cues()
                     if x.get_type() == "light"][0]
            wind = [x for x in treatment.get_initial_cues()
                    if x.get_type() == "wind"][0]

            conflict_light = [x for x in treatment.get_conflict_cues()
                              if x.get_type() == "light"][0]
            conflict_wind = [x for x in treatment.get_conflict_cues()
                             if x.get_type() == "wind"][0]

            # Initialise sample variables
            wind_samples = None
            light_samples = None
            conflict_wind_samples = None
            conflict_light_samples = None

            # If this treatment had preset samples for testing,
            # use those rather than generating fresh ones
            if treatment.preset_samples():
                samples = treatment.get_samples()
                initial = samples["initial"]
                conflict = samples["conflict"]

                wind_samples = initial["wind"]
                light_samples = initial["light"]
                conflict_wind_samples = conflict["wind"]
                conflict_light_samples = conflict["light"]
            else:
                wind_samples = wind.sample(treatment.get_n())
                light_samples = light.sample(treatment.get_n())

                conflict_wind_samples = conflict_wind.sample(treatment.get_n())
                conflict_light_samples = conflict_light.sample(treatment.get_n())

            print("WTA : simulating " + str(treatment.get_n()) + " trials" +
                  " for treatment " + treatment.get_id())

            changes = []
            for trial in range(treatment.get_n()):
                # Get samples for trial
                light_sample = light_samples[trial]
                wind_sample = wind_samples[trial]
                conflict_light_sample = conflict_light_samples[trial]
                conflict_wind_sample = conflict_wind_samples[trial]

                # Retrieve cue weights - these are the kappa parameters
                light_weight_clean = light.get_weight()
                wind_weight_clean = wind.get_weight()
                conflict_light_weight_clean = conflict_light.get_weight()
                conflict_wind_weight_clean = conflict_wind.get_weight()

                # Normalise weights
                light_weight = light_weight_clean / (light_weight_clean + wind_weight_clean)
                wind_weight = wind_weight_clean / (light_weight_clean + wind_weight_clean)
                conflict_light_weight = conflict_light_weight_clean / (conflict_light_weight_clean + conflict_wind_weight_clean)
                conflict_wind_weight = conflict_wind_weight_clean / (conflict_light_weight_clean + conflict_wind_weight_clean)

                # Angular offset between cues has to be taken into account
                offset = wind_sample - light_sample

                # Initial "roll"
                bearing = self.compute_integration(light_sample,
                                              light_weight,
                                              wind_sample,
                                              wind_weight,
                                              offset)

                # Conflict "roll"
                conflict_bearing = self.compute_integration(conflict_light_sample,
                                                       conflict_light_weight,
                                                       conflict_wind_sample,
                                                       conflict_wind_weight,
                                                       offset)

                # Change in bearing
                change = conflict_bearing - bearing

                if self.output():
                    print("Trial #{}".format(treatment.get_n()))
                    print("light_sample angle: " + str(np.degrees(light_sample)))
                    print("light weight: " + str(conflict_light_weight))
                    print("wind_sample angle: " + str(np.degrees(wind_sample)))
                    print("wind weight: " + str(conflict_wind_weight))
                    print("Initial integration: " + str(np.degrees(bearing)))
                    print("Conflict integration: " + str(np.degrees(conflict_bearing)))
                    print("Change: " + str(np.degrees(change)))

                changes.append(change)

            # Store result in treatment object
            treatment.set_changes_in_bearing(changes)
