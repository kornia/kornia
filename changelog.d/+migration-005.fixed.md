`RandomBoxBlur`'s `normalized` argument is documented as what it does. It was described as "if True, L1 norm
of the kernel is set to 1", but it was forwarded positionally into `kornia.filters.box_blur`'s `separable`
parameter: it chooses between the separable and the single 2D pass, and the kernel is L1-normalized either
way, so `normalized=False` returns window means, not sums. The call now passes `border_type` and
`separable` by keyword; outputs are unchanged. (#4433, #4486)
