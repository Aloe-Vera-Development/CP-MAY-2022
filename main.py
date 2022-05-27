from tkinter import Tk
from tkinter.filedialog import askopenfilenames

import eel
from PIL import Image
from PIL.ExifTags import TAGS
from GPSPhoto import gpsphoto
import sqlite3
import shutil

eel.init('web')

def connectToDB():
    conn = sqlite3.connect(r"db.db")
    return conn


def init():
    conn = connectToDB()
    c = conn.cursor()
    c.execute("""CREATE TABLE datas (
        id integer PRIMARY KEY AUTOINCREMENT,
        n integer,
        photo string,
        date datetime,
        lat string,
        long string,
        model string
    )""")


def selectAllData():
    try:
        conn = connectToDB()
        c = conn.cursor()
        recs = c.execute("SELECT * FROM datas")
        for row in recs:
            print(row)
    except:
        init()
        selectAllData()


def selectDataById(id):
    try:
        conn = connectToDB()
        c = conn.cursor()
        recs = c.execute("SELECT * FROM datas WHERE id='" + id + "'")
        for row in recs:
            print(row)
    except:
        init()
        selectDataById(id)


def addData(data):
    try:
        conn = connectToDB()
        c = conn.cursor()
        c.execute("INSERT INTO datas (photo, n, date, lat, long, model) VALUES ('" + data['photo'] + "', '" + str(data['n']) + "', '" + data['date'] + "', '" + data['lat'] + "', '" + data['long'] + "', '" + data['model'] + "')")
        conn.commit()
    except:
        init()


selectAllData()


@eel.expose
def addToDB():
    root = Tk()
    root.withdraw()
    filename = askopenfilenames(filetypes=[("Фотографии", "*.jpg")])
    root.destroy()

    d = {'n': 100}
    for file in filename:
        image = Image.open(file)
        model = ''
        for tag, value in image.getexif().items():
            if TAGS[tag] == 'Make':
                model += value.replace('\x00', '')
            if TAGS[tag] == 'Model':
                model += ' ' + value.replace('\x00', '')
            elif TAGS[tag] == 'DateTime':
                d.update({'date': value})

        d.update({'model': model})

        shutil.copy2(file, 'photos/')
        d.update({'photo': 'photos/'+file.split('/')[-1]})

        data = gpsphoto.getGPSData(file)
        for tag in data.keys():
            if tag == 'Latitude':
                d.update({'lat': str(data[tag])})
            elif tag == 'Longitude':
                d.update({'long': str(data[tag])})

        addData(d)

        print(d)
    return d


eel.start('index.html', size=(1600, 900))
