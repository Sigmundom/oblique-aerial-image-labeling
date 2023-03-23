import shapefile
import utm

with shapefile.Reader("/media/sigmundmestad/aa7e1253-187b-48d4-af32-eb9f050db5dd/sigmunom/Dekningsoversikt/TT-30243_Vertikal-Skrå") as shp:
    shape = shp.shapeRecord(0).shape
    print(shape.points)
    point = utm.from_latlon(*shape.points[0][::-1], force_zone_number=32, force_zone_letter='N')
    print(point)
    print(shp.record(0))
    shapeRec = shp.shapeRecord(0)
    print(shapeRec)
    shape, rec = shp.shapeRecord(0)
    print(shape, rec)