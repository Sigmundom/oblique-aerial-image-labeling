
from libs.sosi import read_sos

def get_image_data(seamline_path, image_name):
    print('Reading seamline file:')
    data = read_sos(seamline_path)

    data_values = data.values()
    # Search through dict for information about current image
    image_data = next((x for x in data_values if 'ImageName' in x and x['ImageName'] == image_name), None)
    return image_data

