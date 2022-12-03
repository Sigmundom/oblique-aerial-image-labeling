# Automatic tiling and annotation for aerial images

## Installation
To install rasterio which is needed for semantic annotation, follow the offical [installation guide](https://rasterio.readthedocs.io/en/latest/installation.html)

## Coordinate systems
In this project there is three different coordinate systems World coordinates(WC), image coordinates(IC) and tile coordinates(TC).

- **World coordinates(WC)**:  (X,Y,Z) = (Easting, Northing, Height)  
    UTM coordinates. 
- **Image coordinates(IC)** - (x,y)   
    Origo in centre of image.  
    Axis: $x\rightarrow y\uparrow$  
- **Tile coordinates(TC)** - (x,y)  
    Origo in the top-left corner.  
    Axis: $x\rightarrow y\downarrow$

