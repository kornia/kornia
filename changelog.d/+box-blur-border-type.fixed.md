`RandomBoxBlur` now accepts the same `border_type` spellings as `RandomGaussianBlur` and
`RandomMotionBlur`: a `BorderType` member, its integer value, or the name in any case. An unknown
name raises `KeyError` at construction instead of failing inside `forward`.
