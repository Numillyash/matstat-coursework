import matplotlib.pyplot as plt

# Чтение данных из файла
with open('var_16_lognorm.csv', 'r') as f:
    data = [float(line.strip()) for line in f.readlines()]

# Построение гистограммы распределения
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(data, bins=range(0, 1000, 1), density=True)

# Настройка внешнего вида графика
ax.set_xlabel('Значение')
ax.set_ylabel('Плотность вероятности')
ax.set_title('Распределение')

# Вывод графика на экран
plt.show()

def dataSum(data):
    s = sum(data)
    return s
print("Data summ is", dataSum(data))

def dataMean(data):
    summ = dataSum(data)
    mean = summ / len(data)
    return mean
print("Data mean is", dataMean(data))

def dataMedian(data):
    dataLen = len(data)
    sortedData = sorted(data)
    if dataLen % 2 == 0:
        # если количество элементов четное, берем среднее двух средних значений
        mid = dataLen // 2
        median = (sortedData[mid-1] + sortedData[mid]) / 2
    else:
        # если количество элементов нечетное, берем среднее значение
        median = sortedData[dataLen//2]
    return median
print("Data median is", dataMedian(data))

def dataModa(data):
    # Создаем словарь, где ключи - элементы списка, а значения - количество их повторений
    countDict = {}
    for i in data:
        if i in countDict:
            countDict[i] += 1
        else:
            countDict[i] = 1

    # Находим элемент(ы) с максимальным количеством повторений
    max_count = max(countDict.values())
    modes = []
    for key, value in countDict.items():
        if value == max_count:
            modes.append(key)

    # Возвращаем моду(ы)
    return modes
print("Data moda is", dataModa(data))

def dataRange(data):
    return max(data) - min(data)
print("Data range is", dataRange(data))

def dataBiasedVariance(data):
    dataLen = len(data)
    mean = dataMean(data)
    squared_differences_sum = sum([(x - mean) ** 2 for x in data])
    return squared_differences_sum / (dataLen)
print("Data Biased Variance is", dataBiasedVariance(data))

def dataUnbiasedVariance(data):
    dataLen = len(data)
    mean = dataMean(data)
    squared_differences_sum = sum([(x - mean) ** 2 for x in data])
    return squared_differences_sum / (dataLen - 1)
print("Data Unbiased Variance is", dataUnbiasedVariance(data))

def dataStartMoment(data, k):
    dataLen = len(data)
    moment = sum([x**k for x in data])/dataLen
    return moment
print("Data start moment is", dataStartMoment(data, 2))

def dataCenterMoment(data, k):
    dataLen = len(data)
    mean = dataMean(data)
    moment = sum([(x - mean) ** k for x in data])/dataLen
    return moment
print("Data start moment is", dataCenterMoment(data, 2))