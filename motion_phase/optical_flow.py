import numpy as np


class OpticalFlow:

    def __init__(self):
        pass


    def calculate_motion(self, prev_regions, curr_regions):

        motion_features = {}

        for region in prev_regions.keys():

            prev_point = np.array(prev_regions[region])
            curr_point = np.array(curr_regions[region])

            # displacement vector
            displacement = curr_point - prev_point

            # speed
            speed = np.linalg.norm(displacement)

            # direction angle
            direction = np.arctan2(
                displacement[1],
                displacement[0]
            )

            motion_features[region] = {

                "dx": displacement[0],
                "dy": displacement[1],

                "speed": speed,

                "direction": direction
            }


        return motion_features