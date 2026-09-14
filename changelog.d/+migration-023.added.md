Every page in the **Models** section of the docs now opens with the shortest script that runs the
model and a figure of its output on a real input, with the paper card moved to the bottom. The
figures are rendered by `docs/generate_model_examples.py` and committed under
`docs/source/_static/img/models/`, so the docs build needs neither the weights nor the sample
images. (#4178)
