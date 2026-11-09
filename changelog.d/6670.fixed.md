`welsch_loss` and `WelschLoss` now accept `reduction=None` and treat it like `"none"`, as `cauchy_loss`, `charbonnier_loss` and `geman_mcclure_loss` already do. It used to raise `BaseError`.
