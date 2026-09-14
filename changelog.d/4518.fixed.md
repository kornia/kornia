The unimplemented tuple-bound `Boxes.clamp`, `Boxes.trim`, and `Boxes.translate(method="fast")` paths now raise
`NotImplementedError` with actionable messages, including the supported tensor bounds and `warp` alternatives where applicable.
