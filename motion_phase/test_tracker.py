from motion_tracker import MotionTracker


tracker = MotionTracker(window_size=5)


motion1 = {
    "mouth":{"speed":2}
}

motion2 = {
    "mouth":{"speed":3}
}

motion3 = {
    "mouth":{"speed":4}
}


tracker.update(motion1)
tracker.update(motion2)

result = tracker.update(motion3)

print(result)