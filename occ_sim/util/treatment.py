"""
treatment.py

Module contains the full class definition for a Treatment.
"""

import numpy as np

class Treatment():
    """
    A Treatement is a specification for a simulation scenario.
    """
    def __init__(self):
        """
        Constructor
        """
        self.__initial_cues = []
        self.__conflict_cues = []
        self.__changes_in_bearing = [] # Simulation output
        self.__mode = "change-in-bearing" # [Legacy] simulation mode
        self.__model = None # Reliability model
        self.__id = None # Unique identifier for a treatment
        self.__n = 0 # Number of individuals
        self.__samples = None # Preset samples if we have any
        self.__preset_samples = False # Preset sample flag

    #
    # Setters
    #
    def set_id(self, ident):
        self.__id = ident

    def set_n(self, n):
        self.__n = n

    def set_mode(self, mode):
        self.__mode = mode

    def set_initial_cues(self, initial_cues):
        self.__initial_cues = initial_cues

    def set_conflict_cues(self, conflict_cues):
        self.__conflict_cues = conflict_cues

    def set_changes_in_bearing(self, changes):
        self.__changes_in_bearing = changes

    def set_reliability_model(self, model):
        self.__model = model

    def generate_samples(self):
        """
        For evaluation purposes it makes more sense to have the same set of
        cue samples which are used for all integration models. The model used
        does not affect the sampling process and including a sampling stage just
        adds randomness to an already variable method of evaluation.

        This function allows the experimenter to generate samples in advance and
        associate them with the treatment, so that each subsequent simulation
        use the pre-generated samples, as opposed to generating their own.
        """
        init_light = [x for x in self.get_initial_cues()
                      if x.get_type() == "light"][0]
        conf_light = [x for x in self.get_conflict_cues()
                      if x.get_type() == "light"][0]

        init_wind = [x for x in self.get_initial_cues()
                     if x.get_type() == "wind"][0]
        conf_wind = [x for x in self.get_conflict_cues()
                     if x.get_type() == "wind"][0]

        wind_samples = init_wind.sample(self.get_n())
        light_samples = init_light.sample(self.get_n())

        conflict_wind_samples = conf_wind.sample(self.get_n())
        conflict_light_samples = conf_light.sample(self.get_n())

        initial = dict({"wind":wind_samples, "light":light_samples})
        conflict = dict({"wind":conflict_wind_samples, "light":conflict_light_samples})

        self.__samples = dict({"initial" : initial, "conflict" : conflict})

        # Flag that this treatment should be run with these samples
        # Useful when comparing integration schemes
        self.__preset_samples = True

        return self.__preset_samples

    #
    # Getters
    #
    def get_samples(self):
        return self.__samples

    def preset_samples(self):
        return self.__preset_samples

    def get_changes_in_bearing(self):
        return self.__changes_in_bearing

    def get_mode(self):
        return self.__mode

    def get_initial_cues(self):
        return self.__initial_cues

    def get_conflict_cues(self):
        return self.__conflict_cues

    def get_light_cues(self):
        init_light = [x for x in self.get_initial_cues()
                      if x.get_type() == "light"][0]
        conf_light = [x for x in self.get_conflict_cues()
                      if x.get_type() == "light"][0]
        return [init_light, conf_light]

    def get_wind_cues(self):
        init_wind = [x for x in self.get_initial_cues()
                     if x.get_type() == "wind"][0]
        conf_wind = [x for x in self.get_conflict_cues()
                     if x.get_type() == "wind"][0]
        return [init_wind, conf_wind]

    def get_n(self):
        return self.__n

    def get_id(self):
        return self.__id

    def get_reliability_model(self):
        return self.__model

    def get_cue_distribution_parameters(self):
        #
        # Extract cue distribution parameters from stored cues
        #
        initial_wind = [x.get_vm_parameters() for x in self.__initial_cues if x.get_type() == "wind"][0]
        initial_light = [x.get_vm_parameters() for x in self.__initial_cues if x.get_type() == "light"][0]
        conflict_wind = [x.get_vm_parameters() for x in self.__conflict_cues if x.get_type() == "wind"][0]
        conflict_light = [x.get_vm_parameters() for x in self.__conflict_cues if x.get_type() == "light"][0]

        initial = [initial_light, initial_wind]
        conflict = [conflict_light, conflict_wind]

        return {'initial': initial, 'conflict': conflict}

    def get_avg_change(self):
        """
        Compute a circular average of the changes we have for this treatment
        """
        angles = self.__changes_in_bearing
        polar_units = list(zip(np.ones(len(angles)), angles))
        cartesian_vecs = [ (r * np.cos(theta), r * np.sin(theta)) for (r, theta) in polar_units]
        x_vals = [ p for (p, _) in cartesian_vecs]
        y_vals = [ p for (_, p) in cartesian_vecs]

        # Take mean
        x_avg = sum(x_vals) / len(x_vals)
        y_avg = sum(y_vals) / len(y_vals)

        r_avg = np.sqrt(x_avg**2 + y_avg**2)
        t_avg = np.arctan2(y_avg, x_avg)

        return r_avg, t_avg

    def validate_treatment(self):
        """
        [Legacy]
        Treatment validation was required for the original implementation which
        used YAML configuration files. This function provided basic treatment
        configuration rules.

        :return: the validitystatus of the treatment.
        """
        # General:
        # 1. You may have a max of one of any type of cue per roll
        # 2. You may not have more than two cues per roll
        initial_types = [x.get_type() for x in self.__initial_cues]
        conflict_types = [x.get_type() for x in self.__conflict_cues]

        initial_dup_invalid = len(set(initial_types)) != len(initial_types)
        conflict_dup_invalid = len(set(conflict_types)) != len(conflict_types)

        if initial_dup_invalid or conflict_dup_invalid:
            return False, "Cue redefined in condition for treatment " + self.__id

        initial_too_long = len(self.__initial_cues) > 2
        conflict_too_long = len(self.__conflict_cues) > 2

        # change-in-bearing:
        # 1. Any cues in the conflict roll must have been present in the initial
        #    roll
        if initial_dup_invalid or conflict_dup_invalid:
            return False, "Too many cues in condition for treatment " + self.__id

        cues_missing = len(self.__initial_cues) == 0 or len(self.__conflict_cues) == 0
        if cues_missing:
            return False, "No cues defined for condition in treatment " + self.__id

        if self.__mode == "change-in-bearing":
            for x in conflict_types:
                if x not in initial_types:
                    return False, "Cue of type " + str(x) + " was not present in initial conditions for treatment " + self.__id

        # For change-in-accuracy mode,
        # 1. The number of cues in either roll need not be equal
        # 2. There must be one cue which is present in both treatments
        elif self.__mode == "change-in-accuracy":
            type_list = initial_types + conflict_types
            type_set = set(type_list)
            if len(type_list) == len(type_set):
                return False, "No overlapping cues in treatment " + self.__id

        return True, ""
