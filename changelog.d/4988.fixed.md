`Boxes.translate` with an unknown `method` now raises `NotImplementedError` with a message that names the accepted
values and repeats the rejected one, instead of an empty message. The exception type is unchanged.
