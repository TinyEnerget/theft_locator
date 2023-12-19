# Программа для обработки данных из csv файла и определения факторов для обучения и взаимодействия с моделью нейронной сети
import pandas as pd
from datetime import datetime
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import warnings
import pickle
import json
warnings.filterwarnings('ignore')


class neural_network:
   
    def __init__(self, file_path):
        self.file_path = file_path
        #self.Initial_df = Initial_df
        #self.Consumption_in_day = Consumption_in_day
        #self.Consumption_in_month = Consumption_in_month
        #self.Factor_UpperLimit = Factor_UpperLimit
        #self.Factor_Negative_consumption = Factor_Negative_consumption
        #self.Factor_owner_type = Factor_owner_type
        #self.Factor_low_consumption_long_period = Factor_low_consumption_long_period

    def Initial_DF_for_process(self, file_path):
        self.file_path = file_path
        Initial_df = pd.read_csv(self.file_path)
        print('ИД Лицевого счета потребителя '+ str(Initial_df.loc[0,'ИД ЛС или Договора']))
        Initial_df['Дата показания'] = pd.to_datetime(Initial_df['Дата показания'], dayfirst=True)
        Initial_df.sort_values(by='Дата показания', inplace=True)
        Initial_df['Показание'] = Initial_df['Показание'].astype(str).str.replace(',','.').astype(float)
        Initial_df['Показание'] = pd.to_numeric(Initial_df['Показание'])
        Initial_df = Initial_df.reset_index(drop=True)

        if len(Initial_df) <= 1:
            print('Выбранный файл содержит менее 2-х строк. Проверьте файл.')
            return None

        return Initial_df

    # Интерполяция данных для выяления месячного потреблния (разница между датами) кусочная интерполяция
    def Interpolation_DF(self,Initial_df):
        self.Initial_df = Initial_df
        Interp_mat = pd.DataFrame()
        days_full = pd.DataFrame()

        for indx in range(len(Initial_df) - 1):

            start_time = pd.to_datetime(self.Initial_df.loc[indx ,['Дата показания']], format='%d.%m.%Y.%H:%M:%S').min()
            end_time = pd.to_datetime(self.Initial_df.loc[(indx + 1) ,['Дата показания']], format='%d.%m.%Y.%H:%M:%S').min()

            start_value = self.Initial_df.loc[indx ,['Показание']]
            end_value = self.Initial_df.loc[(indx + 1) ,['Показание']]

            days = pd.DataFrame({'Day': pd.date_range(start_time, end_time, freq='H')})

            Coms = pd.DataFrame({
                'Дата показания': [start_time, end_time],
                'Показание': [start_value, end_value]
            })

            csp = CubicSpline(Coms['Дата показания'], Coms['Показание'])

            interp_values = csp(days['Day'])

            days_full = pd.concat([days_full, pd.DataFrame(days)], ignore_index=True)
            Interp_mat = pd.concat([Interp_mat, pd.DataFrame(interp_values)], ignore_index=True)

        Consumption_in_day = pd.DataFrame({
        'Date': days_full['Day'],
        'Consumption': Interp_mat[0]
        })

        return Consumption_in_day

    # Функция для построения графиков
    #def Graf_plot(self, Consumption_in_day, Initial_df, Consumption_in_month):

    #    # Отображение графиков показаний
    #    # создание фигуры и осей
    #    fig, ax = plt.subplots() 
    #    # построение 1-го графика  
    #    ax.plot(Consumption_in_month.index, Consumption_in_month['Consumption'], color='red', linestyle = '', marker = 'o', markersize=10)
    #    # построение 2-го графика
    #    ax.plot(Consumption_in_day.index, Consumption_in_day['Consumption'], color='blue', linewidth=2)
    #    # построение 3-го графика
    #    ax.plot(Initial_df['Дата показания'], Initial_df['Показание'], color='green', linewidth=5, linestyle = '--')
    #    # добавление названий осей  
    #    ax.set_xlabel('Дата', fontsize=15)
    #    ax.set_ylabel('Показание', fontsize=15)
    #    ax.set_xticklabels(Consumption_in_month.index.shift(1), rotation=90, fontsize=10)
    #    ax.set_title('График показаний', fontsize=15)
    #    # отображение легенды
    #    ax.legend(['Месячное', 'Суточное интерполированное', 'Исходное']) 
    #    ax.grid(True)
    #    # вывод графика
    #    # создание фигуры и осей
    #    #fig, ax = plt.subplots() 
    #    # Отображение графиков потребления
    #    ax1 = Consumption_in_month.plot(x='Период', y='Потребление', color='red', linewidth=2, marker = 'o', markersize=10)
    #    ax1.set_xlabel('Период', fontsize=15)
    #    ax1.set_ylabel('Потребление', fontsize=15)
    #    ax1.set_title('График потребления', fontsize=15)
    #    ax1.legend(['Месячное потребление'])
    #    ax1.grid(True)
    #    plt.close()
    #    return 


    # Выделение месячного потребления
    def Month_consumption(self, Consumption_in_day, Initial_df):
        Consumption_in_day['Date'] =  pd.to_datetime(Consumption_in_day['Date'])
        Consumption_in_day = Consumption_in_day.set_index('Date')
        Consumption_in_month = Consumption_in_day.resample('M').last()
        Consumption_in_month['Потребление'] = pd.to_numeric(Consumption_in_month['Consumption']) - pd.to_numeric(Consumption_in_month['Consumption']).shift(1)
        Consumption_in_month['Период'] = pd.to_datetime(Consumption_in_month.index).shift(1) - pd.to_datetime(Consumption_in_month.index)

        #neural_network.Graf_plot(Consumption_in_day, Initial_df, Consumption_in_month)

        return Consumption_in_month

    # Определение фактора предельного месячного потребления
    def Factor_of_month_consumption_UpperLimit(self, Consumption_in_month, Upper_limit):

        Factor_UpperLimit = Consumption_in_month['Потребление'].apply(lambda x: 1 if x >= Upper_limit else 0)

        if Factor_UpperLimit.sum() > 0:
            print('Значение месячного потребления свыше ' + str(Upper_limit))
            Factor_UpperLimit = 1
        else:
            print('Значение месячного потребления ниже '+ str(Upper_limit))
            Factor_UpperLimit = 0

        return Factor_UpperLimit

    # Определение фактора отрицательного месячного потребления
    def Factor_of_month_negative_consumption(self, Consumption_in_month):

        Factor_Negative_consumption = Consumption_in_month['Потребление'].apply(lambda x: 1 if x <= 0 else 0).astype(int)

        if Factor_Negative_consumption.sum() > 0:
            print('Значение месячного потребления отрицательно')
            Factor_Negative_consumption = 1
        else:
            print('Значение месячного потребления положительно')
            Factor_Negative_consumption = 0

        return Factor_Negative_consumption

    # Определение фактора Вид владельца
    def Factor_of_owner_type(self, Initial_df):

        if Initial_df.loc[0,['Вид владельца']].values[0] == 'PL':
            print('Владелец - физическое лицо')
            Factor_owner_type = 1
        else:
            print('Владелец - юридическое лицо')
            Factor_owner_type = 0

        return Factor_owner_type

    # Определение фактора большого перерыва между датами показаний
    def Factor_of_long_break(self, Initial_df):

        Break_limit = 50 # Предел перерыва между датами показаний
        Init_DF = pd.DataFrame()
        Init_DF['Период'] = (pd.to_datetime(Initial_df['Дата показания']) - pd.to_datetime(Initial_df['Дата показания']).shift(1)).dt.days
        Factor_long_break = Init_DF['Период'].apply(lambda x: 1 if x >= Break_limit else 0).astype(int)

        if Factor_long_break.sum() > 0:
            print('Большой перерыв между показаниями, максимальный перерыв составил ' + str(Init_DF['Период'].max()) + ' дня')
            Factor_long_break = 1
        else:
            print('Нормальный перерыв между показаниями')
            Factor_long_break = 0

        return Factor_long_break

    # Определение фактора большого перерыва между датами показаний с малым значением потребления
    def Factor_of_low_consumption_long_period(self, Initial_df):

        Break_limit = 50 # Предел перерыва между датами показаний
        Init_DF = pd.DataFrame()
        Init_DF['Период'] = (pd.to_datetime(Initial_df['Дата показания']) - pd.to_datetime(Initial_df['Дата показания']).shift(1)).dt.days
        Init_DF['Потребление'] = ((Initial_df['Показание']) - (Initial_df['Показание']).shift(1))
        Factor_low_consumption_long_period = pd.Series(index=Init_DF.index)

        for indx in range(1, len(Init_DF)):
            if Init_DF.loc[indx,'Период'] >= Break_limit and Init_DF.loc[indx,'Потребление'] < 2000:
                Factor_low_consumption_long_period[indx] = 1
            else:
                Factor_low_consumption_long_period[indx] = 0

        if Factor_low_consumption_long_period.sum() <= 0:
            print('Большой перерыв между показаниями, максимальный перерыв составил ' + str(Init_DF['Период'].max()) 
                      + ' при потреблении ' + str(Init_DF.loc[Init_DF['Период'].idxmax(),'Потребление']))
            Factor_low_consumption_long_period = 1
        else:
            print('Нормальный перерыв между показаниями и при потреблении ниже 2000 кВт')
            Factor_low_consumption_long_period = 0

        return Factor_low_consumption_long_period

    def Create_factors_consumption(self, Factor_Negative_consumption, Factor_owner_type,
                                      Factor_long_break, Factor_UpperLimit, Factor_low_consumption_long_period):
        Factors_consumption = pd.DataFrame({
        'Фактор вид владельца': Factor_owner_type,
        'Фактор большого перерыва между датами показаний': Factor_long_break,
        'Фактор предельного месячного потребления': Factor_UpperLimit,
        'Фактор большого перерыва между датами показаний с малым значением потребления': Factor_low_consumption_long_period,
        'Фактор отрицательного месячного потребления': Factor_Negative_consumption
        }, index = [0])
        print(Factors_consumption, end = '\n\n', flush = True)
        return Factors_consumption

    def neural_network_process(self, Factors_consumption):
        model = pickle.load(open('nfnn_model.pkl', 'rb'))
        Result = model.predict(Factors_consumption)
        if Result == 1:
            Text = 'Данный ЛС подозрительный'
            return Text     
        else:
            Text = 'Данный ЛС не подозрительный'
       
        return Text
        
## Активация функции
#
#file_path = "C:/Users/Vlad Titov/Desktop/results_2ced0a57-f6e7-11e9-80c2-9457a553d5eb.csv"
#
#nn_process = neural_network(file_path)
#
## Инициализация датафрейма
#Initial_df = nn_process.Initial_DF_for_process(file_path)
#
#if Initial_df is None:
#    print('Файл не соответсвует требованиям, должно быть 2 и более показаний.')
#else:
#    # Интерполяция данных
#    Consumption_in_day = nn_process.Interpolation_DF(Initial_df)
#    # Выделение месячного потребления
#    Consumption_in_month = nn_process.Month_consumption(Consumption_in_day, Initial_df)
#    nn_process.Graf_plot(Consumption_in_day, Initial_df, Consumption_in_month)
#    # Определение фактора Вид владельца
#    Factor_owner_type = nn_process.Factor_of_owner_type(Initial_df)
#    # Определение фактора предельного месячного потребления
#    Upper_limit = 2000 # Предел месячного потребления
#    Factor_UpperLimit = nn_process.Factor_of_month_consumption_UpperLimit(Consumption_in_month, Upper_limit)
#    # Определение фактора отрицательного месячного потребления
#    Factor_Negative_consumption = nn_process.Factor_of_month_negative_consumption(Consumption_in_month)
#    # Определение фактора большого перерыва между датами показаний
#    Factor_long_break = nn_process.Factor_of_long_break(Initial_df)
#    # Определение фактора большого перерыва между датами показаний с малым значением потребления
#    Factor_low_consumption_long_period = nn_process.Factor_of_low_consumption_long_period(Initial_df)
#    
#    Factors_consumption = nn_process.Create_factors_consumption(Factor_Negative_consumption,
#                                                                 Factor_owner_type,
#                                                                 Factor_long_break,
#                                                                 Factor_UpperLimit,
#                                                                    Factor_low_consumption_long_period)
#    # Взаимодействие с моделью нейронной сети
#    Text_result = nn_process.neural_network_process(Factors_consumption)
#    Json_str = pd.DataFrame({json.dumps(Text_result)})
#    Json_str.to_json('result.json', orient='records')
#    print(Text_result)
#    
#