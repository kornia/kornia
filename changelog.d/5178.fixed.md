Fix `ImageModule.show()` / `.save()` and core `ImageSequential.show()` / `.save()` after requesting NumPy or PIL output by retaining a detached tensor cache before conversion.
