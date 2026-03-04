# Biomechanics of Piano Playing
<img width="1484" height="835" alt="image" src="https://github.com/user-attachments/assets/485b0b03-8ebf-4309-a6e8-f2c677b92109" />

##  Lens Correction (`lenscor.py`)
The overhead videos are recorded with a Sony FX30 paired with a Sigma LH716-01 f/1.4 DC DN Contemporary Lens. Using the LensFun database, set the distortion parameters to get an output video with no lens warp.

## Warp (`warp.py`)
We only want to evaluate the keyboard region of the overhead videos. The user selects the four approximate corners of the keyboard, and the program automatically refines the corners to output a video with just the keyboard.

## Next Steps
- The keyboard is tilted relative to the camera angle. One of the next steps would be to correct for that.
- Each of the 88 keys need to be mapped so keypresses can be identified. One way to do this would be to use edge detection.
