# liujinhua
First of all, I apologize for the content and code quality of the file I uploaded if it makes you uncomfortable to read.

Please allow me to introduce you to the file I uploaded.
This is a small project based on DCGAN to generate anime face pictures. I apologize for the inability to upload the training data. 
I used about 15,000 training data as the training set. These data are obtained from an anime picture website.
and then store it in the automatically created folder 'imgs'
##The file for obtaining anime pictures is root. py

Then use opencv and other series of operations to extract the face of the entire animation picture
Then all the obtained avatar pictures are stored as a training set in the automatically created folder 'faces'
##The file for operating anime pictures is getfaces. py

Then began to create the GAN model, using tensorflow as the framework, which includes a generator and a discriminator, 
which need to set some hyperparameters and variables, and finally complete the creation of the model
##The file for creating the model is define. py

##The file for training the model is train. py
##The file for generating the pictures is generate. py
