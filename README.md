# Flying Pig I
Please proceed to the `flying_pig` folder.

You will modify the `pig.jpg` image so that it will be misclassified into a plane by a model. (We already provide a pre-trained model to classify images into corresponding classes.)

Your first job is to create an adversarial image by interpolating two images. You can find `pig.jpg` and `plane.jpg`. You can try interpolation of the two images in various ways, e.g., computing the mean of the two images, scaling the plane's pixel values by a factor of two and subtracting the pig data from there, etc.). We provide `template.py` and at the first `TODO`, you can interpolate the two data (pig_data and plane_data). Running the template.py will create `flying_pig.jpg` in the same folder; then, you can run `launcher` and choose option 1 to get the flag.

Good luck.
