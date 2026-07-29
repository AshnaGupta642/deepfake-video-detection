import numpy as np


class ADFMPC:

    def __init__(self):
        pass


    def calculate_speed_features(self, history):

        speeds = []

        for item in history:
            speeds.append(item["speed"])

        speeds = np.array(speeds)

        return {
            "mean_speed": np.mean(speeds),
            "speed_variance": np.var(speeds)
        }


    def calculate_acceleration(self, history):

        speeds = np.array(
            [x["speed"] for x in history]
        )

        if len(speeds) < 3:
            return 0

        velocity_change = np.diff(speeds)

        acceleration = np.mean(
            np.abs(velocity_change)
        )

        return acceleration
    def motion_correlation(self, region1, region2):

        speed1 = np.array(
            [x["speed"] for x in region1]
        )

        speed2 = np.array(
            [x["speed"] for x in region2]
        )


        if len(speed1) < 2 or len(speed2) < 2:
            return 0


        correlation = np.corrcoef(
            speed1,
            speed2
        )[0,1]


        if np.isnan(correlation):
            return 0


        return float(correlation)
    def motion_smoothness(self, history):

        speeds = np.array(
            [x["speed"] for x in history]
        )

        if len(speeds) < 3:
            return 0

        acceleration = np.diff(speeds)

        variance = np.var(acceleration)

        smoothness = np.exp(-variance)

        return float(smoothness)

    def direction_consistency(self, history):

        directions = np.array(
            [
                x["direction"] 
                for x in history
                if "direction" in x
            ]
        )

        if len(directions) < 2:
            return 0

        variance = np.var(directions)

        return 1/(1+variance)
    def region_reliability(self, history):

        smoothness = self.motion_smoothness(history)

        direction = self.direction_consistency(history)


        reliability = (
            0.5 * smoothness +
            0.5 * direction
        )


        return float(reliability)
    

    def calculate_region_correlations(self, motion_history):

        pairs = [

            # Eyebrow - Eye coordination
            ("left_eyebrow", "left_eye"),

            ("right_eyebrow", "right_eye"),


            # Eye - Cheek coordination
            ("left_eye", "left_cheek"),

            ("right_eye", "right_cheek"),


            # Cheek - Lip coordination
            ("left_cheek", "upper_lip"),

            ("right_cheek", "upper_lip"),

            ("left_cheek", "lower_lip"),

            ("right_cheek", "lower_lip"),


            # Lip - Jaw coordination
            ("upper_lip", "lower_lip"),

            ("upper_lip", "jaw"),

            ("lower_lip", "jaw"),


            # Facial symmetry
            ("left_cheek", "right_cheek")

        ]


        correlations = {}


        for r1, r2 in pairs:

            if r1 in motion_history and r2 in motion_history:

                score = self.motion_correlation(
                    motion_history[r1],
                    motion_history[r2]
                )

                correlations[
                    f"{r1}_{r2}_corr"
                ] = score


        return correlations

    def motion_phase_delay(self, region1, region2):

        speed1 = np.array(
            [x["speed"] for x in region1]
        )

        speed2 = np.array(
            [x["speed"] for x in region2]
        )


        if len(speed1) < 2 or len(speed2) < 2:
            return 0


        # normalize
        speed1 = speed1 - np.mean(speed1)
        speed2 = speed2 - np.mean(speed2)


        correlation = np.correlate(
            speed1,
            speed2,
            mode="full"
        )


        delay = np.argmax(correlation) - (len(speed2)-1)


        return int(delay)
    def calculate_phase_delays(self, motion_history):

        pairs = [

            # Eyebrow - Eye coordination
                        ("left_eyebrow", "left_eye"),
            
                        ("right_eyebrow", "right_eye"),
            
            
                        # Eye - Cheek coordination
                        ("left_eye", "left_cheek"),
            
                        ("right_eye", "right_cheek"),
            
            
                        # Cheek - Lip coordination
                        ("left_cheek", "upper_lip"),
            
                        ("right_cheek", "upper_lip"),
            
                        ("left_cheek", "lower_lip"),
            
                        ("right_cheek", "lower_lip"),
            
            
                        # Lip - Jaw coordination
                        ("upper_lip", "lower_lip"),
            
                        ("upper_lip", "jaw"),
            
                        ("lower_lip", "jaw"),
            
            
                        # Facial symmetry
                        ("left_cheek", "right_cheek")
            

        ]


        delays = {}


        for r1, r2 in pairs:

            if r1 in motion_history and r2 in motion_history:

                delay = self.motion_phase_delay(
                    motion_history[r1],
                    motion_history[r2]
                )

                delays[
                    f"{r1}_{r2}_delay"
                ] = delay


        return delays

    def calculate_symmetry_score(self, region1, region2):

        speed1 = np.array(
            [x["speed"] for x in region1]
        )

        speed2 = np.array(
            [x["speed"] for x in region2]
        )


        if len(speed1) == 0 or len(speed2) == 0:
            return 0


        mean1 = np.mean(speed1)
        mean2 = np.mean(speed2)


        symmetry = 1 - (
            abs(mean1 - mean2) /
            (mean1 + mean2 + 1e-6)
        )


        return float(symmetry)

    def calculate_symmetry_features(self, motion_history):

        pairs = [

            ("left_eye", "right_eye"),

            ("left_eyebrow", "right_eyebrow"),

            ("left_cheek", "right_cheek")

        ]


        symmetry_features = {}


        for r1, r2 in pairs:

            if r1 in motion_history and r2 in motion_history:

                score = self.calculate_symmetry_score(
                    motion_history[r1],
                    motion_history[r2]
                )


                symmetry_features[
                    f"{r1}_{r2}_symmetry"
                ] = score


        return symmetry_features
    def calculate_direction_difference(self, region1, region2):

        direction1 = np.array(
            [x["direction"] for x in region1]
        )

        direction2 = np.array(
            [x["direction"] for x in region2]
        )


        if len(direction1) == 0 or len(direction2) == 0:
            return 0


        difference = np.abs(
            direction1 - direction2
        )


        return float(np.mean(difference))
    def calculate_direction_features(self, motion_history):

        pairs = [

            ("upper_lip", "lower_lip"),

            ("upper_lip", "jaw"),

            ("lower_lip", "jaw"),

            ("left_cheek", "right_cheek"),

            ("left_eye", "right_eye"),

            ("left_eyebrow", "right_eyebrow")

        ]


        direction_features = {}


        for r1, r2 in pairs:

            if r1 in motion_history and r2 in motion_history:

                diff = self.calculate_direction_difference(
                    motion_history[r1],
                    motion_history[r2]
                )


                direction_features[
                    f"{r1}_{r2}_direction_difference"
                ] = diff


        return direction_features
    def calculate_rhythm_consistency(self, history):

        speeds = np.array(
            [x["speed"] for x in history]
        )


        if len(speeds) < 3:
            return 0


        # change in speed over time
        speed_changes = np.diff(speeds)


        # consistency of speed changes
        variance = np.var(speed_changes)


        rhythm_score = 1 / (1 + variance)


        return float(rhythm_score)
    def calculate_rhythm_features(self, motion_history):

        rhythm_features = {}


        for region, history in motion_history.items():

            rhythm = self.calculate_rhythm_consistency(
                history
            )


            rhythm_features[
                f"{region}_rhythm_consistency"
            ] = rhythm


        return rhythm_features
    def extract_features(self, motion_history):

        features = {}

        for region, history in motion_history.items():

            speed_features = self.calculate_speed_features(history)

            acceleration = self.calculate_acceleration(history)

            smoothness = self.motion_smoothness(history)

            direction_score = self.direction_consistency(history)

            reliability = self.region_reliability(history)

            features[region] = {

                # Motion speed features
                "mean_speed":
                    float(speed_features["mean_speed"]),

                "speed_variance":
                    float(speed_features["speed_variance"]),


                # Temporal dynamics
                "acceleration":
                    float(acceleration),


                # AD-FMPC consistency features
                "motion_smoothness":
                    float(smoothness),

                "direction_consistency":
                    float(direction_score),

                "region_reliability":
                    float(reliability)

            }
        

        region_corr = self.calculate_region_correlations(
            motion_history
        )
        
        phase_delay = self.calculate_phase_delays(
            motion_history
        )

        features.update(phase_delay)

        features.update(region_corr)
        symmetry = self.calculate_symmetry_features(
            motion_history
        )

        features.update(symmetry)
        direction_features = self.calculate_direction_features(
            motion_history
        )

        features.update(direction_features)
        rhythm_features = self.calculate_rhythm_features(
            motion_history
        )

        features.update(rhythm_features)
        return features