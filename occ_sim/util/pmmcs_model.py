from util.integration_models import Simulator, sigmoid

from scipy.stats import norm
import numpy as np

class BiasedWeightedSum(Simulator):
    """
    My naming here has been a bit poor. Angularly, a weighted sum is equivalent
    to weighted averaging and MurrayMorgenstern-based simulators. This particular
    implementation is to represent a parameterised version of the model which can
    represent a "pure" weighted summation with no bias, but allows the addition
    of bias (drawn from a bias distribution) which is applied within a variable
    window. We can vary both of these parameters to change how bias is applied
    and how bias is distributed which lets us evaluate the likelihood of
    different bias applications.

    Variations on the bias distribution mean are not investigated here, though
    they would have an effect; the assumption is that individuals most likely
    have no bias and if they do it will be evenly distributed between the cues.
    We can change the bias distribution variance which makes it more likely that
    individuals will have biases and makes the potential biases more extreme.

    Variations on the bias window indicate where bias takes effect. A
    bias window with width zero means individual bias is never taken into effect.
    Amaximum bias window measn full bias is always applied. Any other bias window
    means the bias will be scaled depending on how close together the cues are.
    I.e. if the difference between cue weights is large, the effect will be
    small, if the difference is small, the effect will be large.
    """

    def __init__(self, output=False, bias_window=0, bias_variance=0, bias_mu=0):
        """
        Initialise this simulator; default settings make this equivalent to
        MMCS (pure weighted vector summation).
        """
        super().__init__(output)

        self.__bias_mu = bias_mu
        self.__bias_variance = bias_variance
        self.__bias_sigma = np.sqrt(self.__bias_variance)
        self.__bias_window = bias_window
        self.set_name("BiasedWS-{}-{}".format(bias_window, bias_variance))

    def compute_integration(self,
                            light,
                            light_weight,
                            wind,
                            wind_weight,
                            offset):
        # Account for any offset from the initial presentation
        adj_wind = wind - offset

        # Eq. 16 from Murray and Morgenstern, 2010
        l = adj_wind + np.arctan2(
            np.sin(light - adj_wind),
            (wind_weight / light_weight) + np.cos(light - adj_wind))

        return l

    def compute_cue_relationship(self, initial_cues):
       if len(initial_cues) == 1:
           return 0

       wind = [x for x in initial_cues if x.get_type() == "wind"][0]
       light = [x for x in initial_cues if x.get_type() == "light"][0]

       return wind.get_azimuth() - light.get_azimuth()

    def simulate_treatment(self, treatment):
        if self.check_options(treatment):
            light = [x for x in treatment.get_initial_cues()
                     if x.get_type() == "light"][0]
            wind = [x for x in treatment.get_initial_cues()
                    if x.get_type() == "wind"][0]

            conflict_light = [x for x in treatment.get_conflict_cues()
                              if x.get_type() == "light"][0]
            conflict_wind = [x for x in treatment.get_conflict_cues()
                             if x.get_type() == "wind"][0]

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



            # If bias window is 0, then set all biases to zero; there
            # is no window of weight difference in which bias is to be
            # applied.  If bias window is max, scale = 1 for any
            # weight difference; this is somewhat arbitrary, however I
            # wanted to include the case where full bias is included across
            # any weight difference. Maybe only if it's greater than max.

            # If bias variance is zero, then all biases are zero; this is
            # the same as a zero window.

            # Initialise biases to be zero (i.e. no biases)
            biases = np.zeros(treatment.get_n())

            # Draw r.v. samples if bias can be used.
            if self.__bias_sigma != 0 and self.__bias_window != 0:
                print("Biases generated")
                biases = norm.rvs(self.__bias_mu,
                                  self.__bias_sigma,
                                  treatment.get_n())


            print("{}: simulating {} trials for treatment {}".format(
                self.name(),
                treatment.get_n(),
                treatment.get_id()
                )
            )


            changes = []
            for trial in range(treatment.get_n()):
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

                weight_difference = abs(light_weight - wind_weight)

                # Remap inverse differences into 0,0.1 to act as scaling factors
                # One could argue this should be (1 - window) on the denomenator
                # however in practice, the range was better approximated by 0.99
                window = 1 - self.__bias_window
                inv_diff = 1 - weight_difference
                scale = 1 # (inv_diff - window) / (0.99 - window)

                if weight_difference < self.__bias_window:
                    bias = scale * bias
                    print("Bias applied: {}".format(bias))
                else:
                    bias = 0

                # print("W_Diff: {}".format(weight_difference))
                # print("I_Diff: {}".format(inv_diff))
                # print("Bias: {}".format(bias))

                # Include bias
                light_weight = light_weight + bias
                wind_weight = wind_weight - bias
                conflict_light_weight = conflict_light_weight + bias
                conflict_wind_weight = conflict_wind_weight - bias

                # Weight adjustment
                light_weight = sigmoid(light_weight)
                wind_weight = sigmoid(wind_weight)
                conflict_light_weight = sigmoid(conflict_light_weight)
                conflict_wind_weight = sigmoid(conflict_wind_weight)


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

            treatment.set_changes_in_bearing(changes)
