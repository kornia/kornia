Fixed `VideoSequential` raising an `IndexError` for `RandomCutMixV2` with the default `same_on_frame=True`.
The mix parameters are now repeated over the frames along the video axis, so every frame of a video cuts the
same box from the matching frame of the same partner video.
