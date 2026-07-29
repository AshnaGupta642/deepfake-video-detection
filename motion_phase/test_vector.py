from feature_vector import AD_FMPC_Vector
from ad_fmpc import ADFMPC


history = {

    "upper_lip": [
        {
            "speed": 3,
            "direction": 1.2
        },
        {
            "speed": 4,
            "direction": 1.25
        },
        {
            "speed": 5,
            "direction": 1.3
        }
    ],


    "lower_lip": [
        {
            "speed": 3,
            "direction": 1.22
        },
        {
            "speed": 4,
            "direction": 1.27
        },
        {
            "speed": 5,
            "direction": 1.32
        }
    ],


    "jaw": [
        {
            "speed": 3,
            "direction": 1.1
        },
        {
            "speed": 4,
            "direction": 1.15
        },
        {
            "speed": 5,
            "direction": 1.2
        }
    ]

}


model = ADFMPC()

features = model.extract_features(history)


print("AD-FMPC FEATURES")
print(features)


converter = AD_FMPC_Vector()


vector = converter.flatten_features(features)


print("\nFinal AD-FMPC Vector:")
print(vector)


print("\nVector Shape:")
print(vector.shape)


converter.save_vector(
    vector,
    "sample_adfmpc.npy"
)