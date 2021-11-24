"""
[Legacy]

deserialiser.py

This was unused at the end of development but should still work. Input
files can be specified using YAML and then run via main.py but the
available options are quite limited and main.py would need to be modified
for this to work properly. The GUI available via occ_sim.py completely
replaces the functionality available here.
"""

from definitions import CONFIG_FILE
from world.light import Light
from world.wind import Wind

from util.treatment import Treatment

import util.models as models

import yaml
import numpy as np
import sys

class Deserialiser:
    """
    Class to deserialise a pyyaml config file into the objects available.
    """
    def __init__(self, configuration_path=CONFIG_FILE):
        self.__config_path = configuration_path
        self.__master_id_list = []
        self.__reliability_model = models.ReliabilityModel()


    def __decode_optional(self, options, treatment):
        """
        Decodes optional parameters and stores them in the global params module.
        :param doc: The YAML doc for a given treatment.
        :param treatment: the treatment we are currently updating
        :return: unused
        """

        # Set mode of usage, determines legal characteristics of the
        # config file.
        if "mode" in options:
            valid_modes = ["change-in-bearing", "change-in-accuracy"]
            mode = options["mode"]
            treatment.set_mode(mode)
            if mode not in valid_modes:
                sys.exit("Fatal: " + str(mode) + " is not a valid mode.");


    def __decode_treatment_parameters(self,treatment_parameters, treatment):
        """
        Unpack the mandatory treatment parameters.
        :param treatment_parameters: the dictionary containing the treatment
        parameters
        :param treatment: the treatment we are currently updating
        """
        # Error checking
        if "treatment-id" not in treatment_parameters:
            sys.exit("Fatal: missing treatment-id field.")
        elif "number-of-trials" not in treatment_parameters:
            sys.exit("Fatal: missing number-of-parameters field.")
        elif treatment_parameters["treatment-id"] in self.__master_id_list:
            sys.exit("Fatal: multiple definitions of treatment " +
                     str(treatment_parameters["treatment-id"]))

        treatment.set_id(treatment_parameters["treatment-id"])
        treatment.set_n(treatment_parameters["number-of-trials"])

    def __decode_light_cue(self, light, treatment):
        elevation = 0
        azimuth = 0
        if "elevation" not in light:
            sys.exit("Fatal: Light cue missing elevation parameter")
        elif "azimuth" not in light:
            sys.exit("Fatal: Light cue missing azimuth parameter")

        return Light(
            elevation=np.radians(light["elevation"]),
            azimuth=np.radians(light["azimuth"]),
            treatment=treatment
            )

    def __decode_wind_cue(self, wind, treatment):
        speed = 0
        azimuth = 0
        if "speed" not in wind:
            sys.exit("Fatal: Wind cue missing speed parameter")
        elif "azimuth" not in wind:
            sys.exit("Fatal: Wind cue missing azimuth parameter")
        return Wind(speed=wind["speed"],
                    azimuth=np.radians(wind["azimuth"]),
                    treatment=treatment)

    def __decode_initial_cues(self, initial, treatment):
        """
        Decode the initial cue conditions.
        """
        cue_set = []
        if "wind" in initial:
            wind_cue = self.__decode_wind_cue(initial["wind"], treatment)
            cue_set.append(wind_cue)

        if "light" in initial:
            light_cue = self.__decode_light_cue(initial["light"], treatment)
            cue_set.append(light_cue)

        treatment.set_initial_cues(cue_set)

    def __decode_conflict_cues(self, conflict, treatment):
        """
        Decode the conflict conditions and validate the change.
        """
        cue_set = []
        if "wind" in conflict:
            wind_cue = self.__decode_wind_cue(conflict["wind"], treatment)
            cue_set.append(wind_cue)

        if "light" in conflict:
            light_cue = self.__decode_light_cue(conflict["light"], treatment)
            cue_set.append(light_cue)

        treatment.set_conflict_cues(cue_set)

        valid, msg = treatment.validate_treatment()

        if not valid:
            sys.exit("Fatal: " + msg)

    def init_configuration(self):
        """
        Decodes the cues from the config.yaml file and generates the corresponding objects.
        Objects are then saved to a central configuration module.

        :return: A 2D list of the cues for each roll, should use conf directly though.
        """
        with open(self.__config_path) as config:
            docs = yaml.load_all(config, Loader=yaml.FullLoader)
            treatments = []

            for doc in docs:
                treatment = Treatment()
                treatment.set_reliability_model(self.__reliability_model)

                if "optional" in doc:
                    self.__decode_optional(doc['optional'], treatment)

                if "treatment-parameters" in doc:
                    self.__decode_treatment_parameters(
                        doc['treatment-parameters'],
                        treatment)
                else:
                    sys.exit(
                        "Fatal: Missing treatment parameters for a given "
                        "treatment.\n"
                        "Solution: Make sure the treatment-parameters heading "
                        "is defined for all treatments. See example config file."
                    )

                if "initial-condition" in doc:
                    self.__decode_initial_cues(doc["initial-condition"], treatment)
                else:
                    sys.exit("Fatal: missing initial cue conditions for a "
                             "given treatment.\n"
                             "Solution: check all treatments have an "
                             "initial-condition section. See example config "
                             "file."
                    )


                if "conflict-condition" in doc:
                    self.__decode_conflict_cues(doc["conflict-condition"], treatment)
                else:
                    sys.exit("Fatal: missing initial cue conditions for a "
                             "given treatment.\n"
                             "Solution: check all treatments have a "
                             "conflict-condition section. See example config "
                             "file."
                    )

                treatments.append(treatment)

        return treatments
