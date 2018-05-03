This package includes Matlab scripts that implement the proximal operator of the Structure Tensor Total Variation (STV) functional described in the paper:

S. Lefkimmiatis, A. Roussos, P. Maragos, M. Unser, "Structure Tensor Total Variation", SIAM Journal on Imaging Sciences, 2015, in press. 

The run_demo.m file includes an example of the proximal map of STV used for denoising a color image. For the arguments that the proxSTV.m accepts please you can either have a look at the comments inside the proxSTV.m file or type 

>> help proxSTV 

inside the Matlab environment.

The main routine proxSTV depends on the mex script projectSpMat2xNc.c(cpp). In the folder ./mex/src there are precompiled version for Mac OS and Linux. If you wish to run the script proxSTV in a different operating system you will have first to compile the .c/.cpp files. One possible option to do this is by executing the compile.m file located in the ./mex/src subfolder. 

