<h3>Команда Aloe Vera</h3>
<h1 align="center">
  <br>
  <br>
  <br>
  <a href="#">Мой Морж</a>
  <br>
</h1>

<h4 align="center"></h4>

<p align="center">
  <a href="#фичи">Фичи</a> •
  <a href="#стек">Стек</a> •
  <a href="#команда">Команда</a> •
  <a href="#запуск">Запуск</a> •  
  <a href="#архитектура решения">Архитектура</a>
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/78318991/170849394-ddeaf8f7-739f-4e29-88f3-23ed9dd7a3ae.png">
</p>

## Фичи

- База данных для организации информации о моржах.
- Не требует подключения к интернету.
- Автоматический подсчет кол-ва моржей.


## Стек

Этот проект написан с помощью:

- [PyTorch](https://pytorch.org/)
- [eel](https://pypi.org/project/Eel/)
- HTML & CSS + JS
- [sqlite3](https://www.sqlite.org/index.html)
- [Python 3](https://www.python.org/)



## Архитектура решения

Архитектура нейронной сети:

Обученная нами нейронная сеть решает задачу регрессии для определения количества особей на снимке. За основу была взята сеть VGG-16, с некоторыми дополнениями, что позволило улучшить качество регрессии. Модель совершает незначительные отклонения при своей работе. 

![conv-layers-vgg16-3365993888](https://user-images.githubusercontent.com/78318991/170849846-b5980c6c-8c81-4a28-8810-487b34dd4121.jpeg)


<!-- <p align="center">
  <img src="https://user-images.githubusercontent.com/53406289/164958075-1692be7d-3a48-415c-876c-0cf53aadd7f3.png">
</p> -->

База данных:

![Картинка-бд](https://user-images.githubusercontent.com/78318991/170849341-05115aae-8b20-460b-bb39-b174d36004b0.png)


## Запуск

Для запуска решения требуется Python3. 
Чтобы запустить программу, требуется установить зависимости.
```bash
pip install -r requirements.txt
```
После чего можно запустить десктоп-приложение с помощью
```bash
python3 main.py
```


## Команда 

Мой морж разработан командой Aloe Vera в рамках Дальневосточного хакатона 

Aloe Vera - это мы:
- Вера Пуртова - менеджер
- Владислав Осин - бекенд разработчик 
- Вячеслав Иванов - фронтенд разработчик
- Ксения Поалихина - менеджер
- Юлия Козлова - дизайнер


