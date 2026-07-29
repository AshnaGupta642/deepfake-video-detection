import numpy as np


class AD_FMPC_Vector:

    def __init__(self):

        # Fixed region order
        self.regions = [
            "left_eyebrow",
            "right_eyebrow",
            "left_eye",
            "right_eye",
            "left_cheek",
            "right_cheek",
            "upper_lip",
            "lower_lip",
            "jaw"
        ]


        # Fixed feature order inside each region
        self.region_features = [
            "mean_speed",
            "speed_variance",
            "acceleration",
            "motion_smoothness",
            "direction_consistency",
            "region_reliability"
        ]


        # Fixed pair feature order
        self.pair_features = [

            # ---------------- Phase Delay ----------------
            "left_eyebrow_left_eye_delay",
            "right_eyebrow_right_eye_delay",

            "left_eye_left_cheek_delay",
            "right_eye_right_cheek_delay",

            "left_cheek_upper_lip_delay",
            "right_cheek_upper_lip_delay",

            "left_cheek_lower_lip_delay",
            "right_cheek_lower_lip_delay",

            "upper_lip_lower_lip_delay",
            "upper_lip_jaw_delay",
            "lower_lip_jaw_delay",

            "left_cheek_right_cheek_delay",

            # ---------------- Correlation ----------------
            "left_eyebrow_left_eye_corr",
            "right_eyebrow_right_eye_corr",

            "left_eye_left_cheek_corr",
            "right_eye_right_cheek_corr",

            "left_cheek_upper_lip_corr",
            "right_cheek_upper_lip_corr",

            "left_cheek_lower_lip_corr",
            "right_cheek_lower_lip_corr",

            "upper_lip_lower_lip_corr",
            "upper_lip_jaw_corr",
            "lower_lip_jaw_corr",

            "left_cheek_right_cheek_corr",

            # ---------------- Direction Difference ----------------
            "upper_lip_lower_lip_direction_difference",
            "upper_lip_jaw_direction_difference",
            "lower_lip_jaw_direction_difference",

            "left_cheek_right_cheek_direction_difference",
            "left_eye_right_eye_direction_difference",
            "left_eyebrow_right_eyebrow_direction_difference",

            # ---------------- Symmetry ----------------
            "left_eye_right_eye_symmetry",
            "left_eyebrow_right_eyebrow_symmetry",
            "left_cheek_right_cheek_symmetry",

            # ---------------- Rhythm ----------------
            "left_eyebrow_rhythm_consistency",
            "right_eyebrow_rhythm_consistency",

            "left_eye_rhythm_consistency",
            "right_eye_rhythm_consistency",

            "nose_rhythm_consistency",

            "left_cheek_rhythm_consistency",
            "right_cheek_rhythm_consistency",

            "upper_lip_rhythm_consistency",
            "lower_lip_rhythm_consistency",

            "jaw_rhythm_consistency"
        ]


    def flatten_features(self, features):

        vector = []


        # 1. Region features
        for region in self.regions:

            if region in features:

                for feat in self.region_features:

                    if feat in features[region]:

                        vector.append(
                            float(features[region][feat])
                        )

                    else:

                        # missing feature
                        vector.append(0.0)


            else:

                # missing region
                vector.extend(
                    [0.0] * len(self.region_features)
                )


        # 2. Pair features
        for feat in self.pair_features:

            if feat in features:

                vector.append(
                    float(features[feat])
                )

            else:

                vector.append(0.0)


        return np.array(
            vector,
            dtype=np.float32
        )


    def save_vector(self, vector, path):

        np.save(path, vector)


    def load_vector(self, path):

        return np.load(path)