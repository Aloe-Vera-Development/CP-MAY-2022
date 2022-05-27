from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import eel
from PIL import Image
from PIL import ExifTags
from GPSPhoto import gpsphoto

eel.init('web')


@eel.expose
def pythonFunction():
    root = Tk()
    root.withdraw()
    filename = askopenfilenames(filetypes=[("Фотографии", "*.jpg")])
    root.quit()

    datas = []
    for file in filename:
        data = gpsphoto.getGPSData(file)
        gps = []
        for tag in data.keys():
            if tag == 'Latitude' or tag == 'Longitude':
                gps.append(data[tag])
        datas.append(gps)
    return datas


eel.start('index.html', size=(1600, 900))
