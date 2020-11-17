# MicroLens-3D-Reconstruction-with-Blender

# 2D simulation

1. Check Camera Info

[Phantom VEO-E 340L](https://www.phantomhighspeed.com/products/cameras/veo/veoe340l) with [DataSheet](./wDSVEOE.pdf)

Sensor Size 25.6 X 16 mm at full resolution. 

Pixel Pitch 10 μm 

2. Check MLA Info

Micro Lens array: [MLA-S125-f30](https://www.rpcphotonics.com/product/mla-s125-f30/) with [DataSheet](./Fact-Sheet-MLA1.pdf)

![f/#](https://github.com/lipilian/MicroLens-3D-Reconstruction-with-Blender/blob/master/Fnumber.JPG). f/# 30 

Micro Lens Pitch: 125 μm 

Focal Length: 125 * 30 = 3750 μm  = 3.75mm

Micro lens principal plane spacing(H1s, H2s): **we don't know**, assume 200 μm 

3. Simulate with 60mm focual length camera

![IMAGE](https://github.com/lipilian/MicroLens-3D-Reconstruction-with-Blender/blob/master/2D1.JPG)


# Check Blender folder CamGen Model.

1. Fork from [Github](https://github.com/Arne-Petersen/Plenoptic-Simulation) and flow the installation guide.

2. I use Windows. Chech [add-on folder](https://docs.blender.org/manual/en/latest/advanced/blender_directory_layout.html#platform-dependent-paths)

# 3D refocuse

## MatLab Tool Box LFIT GUI

1. Camera parameters and sample image:
![sample image about chessboard](https://github.com/lipilian/MicroLens-3D-Reconstruction-with-Blender/blob/master/Image/Check.jpg)



## CMake compile the data


## MatPIV for 3D velocity calculation