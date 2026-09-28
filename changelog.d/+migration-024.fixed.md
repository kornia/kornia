`Boxes.to_mask` leaves list-padding channels empty, including after coordinate transforms,
instead of treating padding entries as real boxes. (#4390)
