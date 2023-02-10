from enum import Enum

class SurfaceType(Enum):  
    ROOF = 1
    WALL = 2
    TERRACE = 3
    TERRACE_WALL = 4
    AUTO_GENERATED_HANDRAIL = 5


    @classmethod
    def parse(cls, type_string):
        if type_string == 'RoofSurface':
            return cls.ROOF
        elif type_string == 'WallSurface':
            return cls.WALL
        else:
            raise Exception(f'Surface type "{type_string}" not recoginized')

if __name__ == '__main__':
    a = SurfaceType('roof')