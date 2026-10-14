`guided_blur` and `GuidedBlur` convert integer and bool images to float32 before covariance products, avoiding wrapped products and dtype errors while preserving the original image scale.
