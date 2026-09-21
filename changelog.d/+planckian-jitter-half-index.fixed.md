PlanckianJitterGenerator keeps its index sampler in float32 when the augmentation is moved to float16 or bfloat16, so the sampled Planckian table index cannot go one past the end. (#4553)
