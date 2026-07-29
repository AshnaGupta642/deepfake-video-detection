import numpy as np


class MotionTracker:

    def __init__(self, window_size=5):

        self.window_size = window_size
        self.history = {}


    def update(self, motion):

        for region, data in motion.items():

            if region not in self.history:
                self.history[region] = []

            self.history[region].append(data)


            # keep only last 5 frames
            if len(self.history[region]) > self.window_size:
                self.history[region].pop(0)


        return self.history