camtoworld_to_worldtocam_Rt` and `worldtocam_to_camtoworld_Rt` accept `check_rotation=True`, which raises `ValueError` when `R` is not a rotation matrix instead of silently returning a wrong pose.
