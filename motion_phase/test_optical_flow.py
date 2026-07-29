from optical_flow import OpticalFlow


prev = {

    "left_eye": (100,100),
    "right_eye": (150,100),
    "mouth": (125,150),
    "jaw": (125,180)

}


curr = {

    "left_eye": (102,101),
    "right_eye": (152,102),
    "mouth": (125,155),
    "jaw": (125,184)

}


flow = OpticalFlow()

motion = flow.calculate_motion(
    prev,
    curr
)


print(motion)