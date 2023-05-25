from PIL import Image

def evaluate():
    nadir_path = 'compiled/nadir_sharp.png'
    combined_path = 'compiled/combined_1_75.png'

    nadir_im = Image.open(nadir_path)
    combined_im = Image.open(combined_path)

    diff = nadir_im - combined_im

    print(diff)

if __name__ == '__main__':
    evaluate()