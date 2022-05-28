from tkinter import Tk
from tkinter.filedialog import askopenfilenames

import eel
from PIL import Image
from PIL.ExifTags import TAGS
from GPSPhoto import gpsphoto
import sqlite3

eel.init('web')


def connectToDB():
    conn = sqlite3.connect(r"db.db")
    return conn


def initDatas():
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


def initStats():
    conn = connectToDB()
    c = conn.cursor()
    c.execute("""CREATE TABLE stats (
        id integer PRIMARY KEY AUTOINCREMENT,
        date datetime,
        value string,
        rookery string
    );""")


def initRookery():
    conn = connectToDB()
    c = conn.cursor()
    c.execute("""CREATE TABLE rookery (
        id integer PRIMARY KEY AUTOINCREMENT,
        name string
    )""")


def getAllData(last=50):
    try:
        conn = connectToDB()
        c = conn.cursor()
        recs = c.execute("SELECT * FROM datas ORDER BY id DESC LIMIT " + str(last))
        rows = []
        for row in recs:
            rows.append(row)
        return rows
    except:
        initDatas()
        getAllData()


def getDataById(id):
    try:
        conn = connectToDB()
        c = conn.cursor()
        recs = c.execute("SELECT * FROM datas WHERE id='" + id + "'")
        for row in recs:
            print(row)
    except:
        initDatas()
        getAllData(id)


def addData(data):
    try:
        conn = connectToDB()
        c = conn.cursor()
        c.execute("INSERT INTO datas (photo, n, date, lat, long, model) VALUES ('" + data['photo'] + "', '" + str(
            data['n']) + "', '" + data['date'] + "', '" + data['lat'] + "', '" + data['long'] + "', '" + data[
                      'model'] + "')")
        conn.commit()
    except:
        initDatas()
        addData(data)


def addRookery(name):
    try:
        conn = connectToDB()
        c = conn.cursor()
        c.execute("INSERT INTO rookery (name) VALUES ('" + name + "')")
        conn.commit()
    except:
        initRookery()
        addRookery(name)


def addStat(date, value, rookery):
    try:
        conn = connectToDB()
        c = conn.cursor()
        c.execute(
            "INSERT INTO stats (date, value, rookery) VALUES ('" + date + "', '" + value + "', '" + rookery + "')")
        conn.commit()
    except:
        initStats()
        addStat(date, value, rookery)


def getAllStat():
    try:
        conn = connectToDB()
        c = conn.cursor()
        recs = c.execute("SELECT * FROM rookery")
        rookeries = []
        for row in recs:
            rookeries.append(row)

        recs = c.execute("SELECT * FROM stats")
        rows = []
        for row in recs:
            rows.append([row[0], row[1], row[2], rookeries[row[3]][1]])
        return rows
    except:
        initStats()
        getAllStat()


@eel.expose
def addToDB():
    root = Tk()
    root.withdraw()
    filename = askopenfilenames(filetypes=[("Фотографии", "*.jpg")])
    root.destroy()

    ans = []

    for file in filename:
        d = {'n': 100}
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

        image.save('web/photos/' + file.split('/')[-1], quality=10)

        d.update({'photo': 'photos/' + file.split('/')[-1]})

        data = gpsphoto.getGPSData(file)
        for tag in data.keys():
            if tag == 'Latitude':
                d.update({'lat': str(data[tag])})
            elif tag == 'Longitude':
                d.update({'long': str(data[tag])})

        addData(d)

        ans.append(d)

        print(d)
    return ans


@eel.expose
def getLastData():
    datas = getAllData()
    print(datas)
    return datas

# addRookery('Крутое лежбище')
#
# addStat('12.12.2021', '5', '0')
# addStat('12.12.2022', '6', '0')

# print(getAllStat())

eel.start('index.html', size=(1600, 900))
