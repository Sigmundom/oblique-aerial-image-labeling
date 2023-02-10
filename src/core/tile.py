import shapely.geometry as sg
from shapely.affinity import affine_transform

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiled_image import TiledImage


class Tile():
    def __init__(self, parent: "TiledImage", crop_box: tuple[int, int, int, int], bbox: sg.Polygon):
        self.parent = parent
        self.crop_box = crop_box # Pixel coordinates (left, upper, right, lower)
        self.bbox = bbox # Image coordinates

    def __repr__(self):
        return f'{self.parent.image_name}_{"_".join(str(x) for x in self.crop_box)}'

    @property
    def tile_image(self):
        return self.parent.image.crop(self.crop_box)

    @property
    def bounds(self):
        """Returns the bounds of the tile in image coordinates (minx, miny, maxx, maxy)."""
        return self.bbox.bounds

    def ic_to_tc(self, polygon_ic: sg.Polygon) -> sg.Polygon:
        xoff, yoff, _, _ = self.bounds
        matrix =    (1.0, 0.0, 0.0,
                    0.0, -1.0, 0.0,
                    0.0, 0.0, 1.0,
                    -xoff, yoff+512, 0.0)
        return affine_transform(polygon_ic, matrix)



    def save(self) -> None:
        self.tile_image.save(f'{self.parent.output_folder}/images/{repr(self)}.jpg')