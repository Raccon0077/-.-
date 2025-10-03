import numpy as np
import pandas as pd
from matplotlib import pyplot as plt





data = pd.read_csv('titanic_train.csv',
                  index_col='PassengerId')

# Первые 5 строк
data.head(5)


data.describe()



# Давайте выберем тех пассажиров, которые отправились в Cherbourg (Embarked = C) и заплатили > 200 фунтов за билет (fare > 200).
data[(data['Embarked'] == 'C') & (data.Fare > 200)].head()



data[(data['Embarked'] == 'C') &
     (data['Fare'] > 200)].sort_values(by='Fare',
                               ascending=False).head()


# Давайте добавим новый признак.
def age_category(age):
    '''
    < 30 -> 1
    >= 30, <55 -> 2
    >= 55 -> 3
    '''
    if age < 30:
        return 1
    elif age < 55:
        return 2
    elif age >= 55:
        return 3

    age_categories = [age_category(age) for age in data.Age]
    data['Age_category'] = age_categories


# Задача 1: Сколько мужчин/женщин было на борту?
gender = data['Sex'].value_counts()
print("1. Сколько мужчин/женщин было на борту?")
print(f"Мужчины: {gender['male']} \nЖенщины: {gender['female']}")
print()


# Задача 2: Распределение Pclass по полу
print("2. Сколько людей из второго класса было на борту? Для мужчин и женщин отдельно.")
pclass_gender = data.groupby(['Pclass', 'Sex']).size()
print(f"Людей во втором классе: {pclass_gender[2].sum()}")
print()



# Задача 3: Медиана и стандартное отклонение Fare
median_fare = round(data['Fare'].median(), 2)
std_fare = round(data['Fare'].std(), 2)
print("3. Каковы медиана и стандартное отклонениеFare?. Округлите до 2-х знаков после запятой.")
print(f"Медиана: {median_fare}, STD: {std_fare}")
print()


# Задача 4: Сравнение возраста выживших и погибших
survived_age = data.groupby('Survived')['Age'].mean()
print("4. Правда ли, что средний возраст выживших людей выше, чем у пассажиров, которые в конечном итоге умерли?")
print("Средний возраст выживших выше?" if survived_age[1] > survived_age[0] else "Нет")
print()


# Задача 5: Выживаемость по возрастным группам
young = data[data['Age'] < 30]['Survived'].mean()
old = data[data['Age'] > 60]['Survived'].mean()
print("5. Это правда, что пассажиры моложе 30 лет. выжили чаще, чем те, кому больше 60 лет. Каковы доли выживших людей среди молодых и пожилых людей?")
print(f"Молодые: {young:.1%} \nПожилые: {old:.1%}")
print()


# Задача 6: Выживаемость по полу
male_surv = data[data['Sex'] == 'male']['Survived'].mean()
female_surv = data[data['Sex'] == 'female']['Survived'].mean()
print("6. Правда ли, что женщины выживали чаще мужчин? Каковы доли выживших людей среди мужчин и женщин?")
print(f"Мужчины: {male_surv:.1%} \nЖенщины: {female_surv:.1%}")
print()


# Задача 7: Самое популярное мужское имя
# Извлекаем имена из полного имени
male_names = data[data['Sex'] == 'male']['Name'].str.extract(' (Mr\. |Master\. |Miss\. |Mrs\. )?([A-Za-z]+)')[1]
print("7. Какое имя наиболее популярно среди пассажиров мужского пола?")
print(f"Самое популярное имя: {male_names.mode().values[0]}")
print()


# Настройки для отображения
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
pd.set_option("display.precision", 2)

# Загрузка данных
data = pd.read_csv('titanic_train.csv', index_col='PassengerId')

# Задача 8: Как средний возраст мужчин/женщин зависит от Pclass?
print("8. Как средний возраст мужчин / женщин зависит от Pclass? Выберите все правильные утверждения.")
print()

# Группируем данные по классу и полу, вычисляем средний возраст
age_by_class_sex = data.groupby(['Pclass', 'Sex'])['Age'].mean().unstack()
print("\nСредний возраст пассажиров по классам и полу:")
print(age_by_class_sex)
print()

# Проверяем утверждения из задания
print("Проверка утверждений:")
print()

# Утверждение 1: В среднем мужчины 1 класса старше 40 лет
men_first_class_avg_age = age_by_class_sex.loc[1, 'male']
print(f"1. Мужчины 1 класса в среднем старше 40 лет ({men_first_class_avg_age:.1f} лет)? {men_first_class_avg_age > 40}")

# Утверждение 2: В среднем женщины 1 класса старше 40 лет
women_first_class_avg_age = age_by_class_sex.loc[1, 'female']
print(f"2. Женщины 1 класса в среднем старше 40 лет ({women_first_class_avg_age:.1f} лет)? {women_first_class_avg_age > 40}")

# Утверждение 3: Мужчины всех классов в среднем старше, чем женщины того же класса
men_older = True
for pclass in [1, 2, 3]:
    if age_by_class_sex.loc[pclass, 'male'] <= age_by_class_sex.loc[pclass, 'female']:
        men_older = False
        break
print(f"3. Мужчины всех классов старше женщин того же класса? {men_older}")

# Утверждение 4: Пассажиры 1 класса старше, чем 2-го, которые старше, чем 3-го
class_order = True
for sex in ['male', 'female']:
    if not (age_by_class_sex.loc[1, sex] > age_by_class_sex.loc[2, sex] > age_by_class_sex.loc[3, sex]):
        class_order = False
        break
print(f"4. Пассажиры 1 класса старше 2-го, которые старше 3-го? {class_order}")

# Визуализация результатов
fig, (ax1) = plt.subplots(1, figsize=(7, 7))

# График: Средний возраст по классам и полу
age_by_class_sex.plot(kind='bar', ax=ax1)
ax1.set_title('Средний возраст пассажиров по классам и полу')
ax1.set_xlabel('Класс')
ax1.set_ylabel('Средний возраст')
ax1.legend(title='Пол')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# Дополнительная статистика
print("\nДополнительная статистика:")
print()
print("Количество пассажиров по классам и полу:")
print(data.groupby(['Pclass', 'Sex']).size().unstack())