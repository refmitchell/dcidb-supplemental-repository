"""
[Experimental]

parallax_model.py

This module was an attempt to create a simulator which included the 3D
spatial relationship of the cues and the changes that an individual would
experience as they moved through the arena.

This simulator works, however it is slow and likely incredibly difficult
to evaluate against the dataset. Definitely warrants further investigation,
however it was out of scope for the current work.
"""

from util.integration_models import Simulator, sigmoid
from util.vec3 import *
from util.treatment import Treatment
from util.models import ReliabilityModel

from world.light import Light

import pdb

class GeometricSimulator(Simulator):
    """
    Experimental effort to model the problem in 3D with a route
    as opposed to exit angle.

    Want to see how the changing spatial relationship of the cues cna affect
    the agents as they navigate and affect the resultant exit angles.
    """

    def __init__(self, output=False):
        super().__init__(output)
        self.set_name("GS")

    def compute_integration(self,
                            light,
                            light_weight,
                            wind,
                            wind_weight,
                            offset=0):
        # Account for any offset from the initial presentation
        adj_wind = wind - offset
#        wind_weight=0
        # Eq. 16 from Murray and Morgenstern, 2010
        l = adj_wind + np.arctan2(
            np.sin(light - adj_wind),
            (wind_weight / light_weight) + np.cos(light - adj_wind))

        return l

    def simulate_treatment(self, treatment):
        """
        3D treatment simulation. We assume wind does not experience any parallax
        effect as the agent moves. Want to simulate two runs from each individual

        :param treatment: The initial conditions of the treatment to be simulated
        :return: A set of paths for each individual
        """

        # Light arch radius in m
        dome_radius = 1.5

        # Arena radius, 30cm
        arena_radius = 0.3

        # Step size in m
        step_size = 0.001

        # Arena centre is treated as our origin
        origin = Vec3(magnitude=0, theta=0, phi=0)

        # Current positions for initial and conflict conditions.
        current = [Vec3(magnitude=0, theta=0, phi=0),
                   Vec3(magnitude=0, theta=0, phi=0)]

        # Origin -> Current
        OC = [ vector_sum(origin, c) for c in current ]

        # Previous -> Current
        PC = [ vector_sum(origin, c) for c in current ]

        # Construct 3D light representation
        light_cues = treatment.get_light_cues()

        light = [ Vec3(magnitude=dome_radius,
                       theta=light_cues[0].get_elevation(),
                       phi=light_cues[0].get_azimuth()),
                  Vec3(magnitude=dome_radius,
                       theta=light_cues[1].get_elevation(),
                       phi=light_cues[1].get_azimuth()) ]

        wind_cues = treatment.get_wind_cues()

        init_paths = []
        conf_paths = []

        changes = []

        # For each individual, simulate both runs simultaneously
        for trial in range(treatment.get_n()):

            # Chosen direction is uniform randomly selected between
            # 0, 360 with resolution of 1 degree
            direction = np.random.random_integers(0,360)

            # Exit angles for each run
            exit_angles = [-1, -1]

            # Sample cues for the first step
            light_samples = [ x.sample() for x in light_cues ]
            wind_samples = [ x.sample() for x in wind_cues ]

            # Offset is the difference between initial samples
            offset = wind_samples[0] - light_samples[0]

            # Initial and conflict paths as coordinate sequences.
            init_path = []
            conf_path = []

            while OC[0].get_magnitude() < arena_radius:
                # Compute weights - start with optimal
                wind_weights = [ x.get_weight() for x in wind_cues ]
                light_weights = [ x.get_weight() for x in light_cues ]
                print("L: {}".format(light_weights))
                print("W: {}".format(wind_weights))
#                input()

                # Compute integrated cues for each roll
                integrated = [
                    self.compute_integration(light_samples[0],
                                             light_weights[0],
                                             wind_samples[0],
                                             wind_weights[0],
                                             offset
                    ),
                    self.compute_integration(light_samples[1],
                                             light_weights[1],
                                             wind_samples[1],
                                             wind_weights[1],
                                             offset
                    )
                ]

#                pdb.set_trace()#

                # Direction of travel is sample integration + (random) chosen
                # bearing mod 2pi.
                step_angles = [ (x + np.radians(direction)) % 2*np.pi
                                for x in integrated ]

                step_vecs = []
                for a in step_angles:
                    step_vecs.append(Vec3(magnitude=step_size,
                                          theta=0,
                                          phi=a))


                # step_vecs = [ Vec3(magnitude=step_size,
                #                    theta=0,
                #                    phi=x) for x in step_angles
                #]

                # Store current position as x,y,z

                init_path.append(current[0].get_polar_as_list())
                conf_path.append(current[1].get_polar_as_list())

                # Update position
                current = [vector_sum(a,b) for (a,b) in zip(current, step_vecs)]
                OC = [ vector_sum(origin, c) for c in current ]

                # Compute new light position relative to current agent position
                CL = [ define(c, l) for (c,l) in zip(current, light) ]

                # Redefine light cues w.r.t. current location
                light_cues = [ Light(elevation=CL[0].get_spherical_as_list()[1],
                                     azimuth=CL[0].get_spherical_as_list()[2],
                                     treatment=treatment),
                               Light(elevation=CL[1].get_spherical_as_list()[1],
                                     azimuth=CL[1].get_spherical_as_list()[2],
                                     treatment=treatment) ]

                # Sample cues for the next step
                light_samples = [ x.sample() for x in light_cues ]
                wind_samples = [ x.sample() for x in wind_cues ]

            # Store path for this beetle
            init_paths.append(init_path)
            conf_paths.append(conf_path)

            change = OC[1].get_spherical_as_list()[2] - OC[0].get_spherical_as_list()[2]
            changes.append(change)

        paths = [ init_paths, conf_paths ]
        treatment.set_changes_in_bearing(changes)
        return paths

