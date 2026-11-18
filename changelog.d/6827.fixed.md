`KMeans.fit` releases completed distance and assignment buffers before later assignments, reducing temporary tensor memory in eager fitting without changing centroid calculations.
