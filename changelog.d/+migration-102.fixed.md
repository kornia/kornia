`VisualPrompter.predict()` called without any prompt (the "run the prediction without prompts" example on the
Segment Anything page) raised `TypeError: object of type 'NoneType' has no len()` from the prompt augmentation
container; it now predicts from the image embedding alone, as documented. (#4178)
