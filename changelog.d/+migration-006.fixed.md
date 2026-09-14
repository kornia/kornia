`So3.exp`, `So3.log`, `Se3.exp`, `Se3.log` and `Se2.exp` return finite gradients at the identity,
and `So3.log` also at a half turn, instead of `nan` on every dtype. The forward values are unchanged;
`torch.where` was differentiating the singular branch it does not select. (#4405)
