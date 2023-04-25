# Цель
  Целью этой работы является создание программной модели искусственной нейронной сети прямого распространения сигналов с использованием библиотек TensorFlow и Keras. А также сравнить влияние разных оптимизаторов и их параметров на процесс обучения. Поиск сетевой архитектуры с наивысшей точностью классификации.
# Язык программирования
  Python 3.10
# Библиотеки 
  Tensorflow,
  Keras,
  Numpy,
  Matplotlib,
  PIL.
# Полученные результаты
Adam,lr=0.001
313/313 [==============================] - 2s 5ms/step - loss: 0.0717 - accuracy: 0.9782
Accuracy: 0.9782000184059143
---------------------       
Adam,lr=0.01
313/313 [==============================] - 5s 14ms/step - loss: 0.1265 - accuracy: 0.9752 
Accuracy: 0.9751999974250793
---------------------
Adam,lr=0.1
313/313 [==============================] - 4s 13ms/step - loss: 0.5527 - accuracy: 0.8983
Accuracy: 0.8982999920845032
---------------------       
Adagrad,lr=0.001
313/313 [==============================] - 3s 7ms/step - loss: 0.5316 - accuracy: 0.8729
Accuracy: 0.8729000091552734
---------------------       
Adagrad,lr=0.01
313/313 [==============================] - 3s 8ms/step - loss: 0.2402 - accuracy: 0.9320
Accuracy: 0.9319999814033508
---------------------
Adagrad,lr=0.1
313/313 [==============================] - 3s 9ms/step - loss: 0.0820 - accuracy: 0.9744
Accuracy: 0.974399983882904
---------------------
Nadam,lr=0.001
313/313 [==============================] - 2s 7ms/step - loss: 0.0691 - accuracy: 0.9783
Accuracy: 0.9782999753952026
---------------------
Nadam,lr=0.01
313/313 [==============================] - 2s 6ms/step - loss: 0.1371 - accuracy: 0.9741
Accuracy: 0.9740999937057495
---------------------
Nadam,lr=0.1
313/313 [==============================] - 2s 7ms/step - loss: 0.4413 - accuracy: 0.9193
Accuracy: 0.9193000197410583
---------------------
dict_values([0.9782000184059143, 0.9751999974250793, 0.8982999920845032, 0.8729000091552734, 0.9319999814033508, 0.974399983882904, 0.9782999753952026, 0.9740999937057495, 0.9193000197410583])

Testing best model with optimizer Nadam,lr=0.001 on custom images:

1/1 [==============================] - 0s 391ms/step
Loaded image : [0]
---------------------
1/1 [==============================] - 0s 31ms/step
Loaded image : [1]
---------------------
1/1 [==============================] - 0s 16ms/step
Loaded image : [2]
---------------------
1/1 [==============================] - 0s 112ms/step
Loaded image : [3]
---------------------
1/1 [==============================] - 0s 24ms/step
Loaded image : [4]
---------------------
1/1 [==============================] - 0s 31ms/step
Loaded image : [5]
---------------------
1/1 [==============================] - 0s 31ms/step
Loaded image : [6]
---------------------
1/1 [==============================] - 0s 16ms/step
Loaded image : [8]
---------------------
