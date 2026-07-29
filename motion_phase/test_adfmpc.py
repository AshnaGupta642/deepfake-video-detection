from ad_fmpc import ADFMPC


history = {

"left_eyebrow":[
 {"speed":1,"direction":0.4},
 {"speed":2,"direction":0.5},
 {"speed":2,"direction":0.45},
 {"speed":3,"direction":0.48}
],

"right_eyebrow":[
 {"speed":1,"direction":0.42},
 {"speed":2,"direction":0.52},
 {"speed":2,"direction":0.47},
 {"speed":3,"direction":0.5}
],


"left_eye":[
 {"speed":2,"direction":0.6},
 {"speed":3,"direction":0.65},
 {"speed":3,"direction":0.62},
 {"speed":4,"direction":0.64}
],


"right_eye":[
 {"speed":2,"direction":0.61},
 {"speed":3,"direction":0.66},
 {"speed":3,"direction":0.63},
 {"speed":4,"direction":0.65}
],


"left_cheek":[
 {"speed":1,"direction":0.8},
 {"speed":2,"direction":0.82},
 {"speed":2,"direction":0.81},
 {"speed":3,"direction":0.83}
],


"right_cheek":[
 {"speed":1,"direction":0.79},
 {"speed":2,"direction":0.81},
 {"speed":2,"direction":0.80},
 {"speed":3,"direction":0.82}
],


"upper_lip":[
    {"speed":2,"direction":1.2},
    {"speed":3,"direction":1.25},
    {"speed":4,"direction":1.22},
    {"speed":5,"direction":1.24}
],


"lower_lip":[
    {"speed":2,"direction":1.15},
    {"speed":3,"direction":1.20},
    {"speed":4,"direction":1.18},
    {"speed":5,"direction":1.21}
],


"jaw":[
 {"speed":2,"direction":1.1},
 {"speed":3,"direction":1.15},
 {"speed":4,"direction":1.12},
 {"speed":5,"direction":1.14}
]

}

model = ADFMPC()

features = model.extract_features(history)

print(features)