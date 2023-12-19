from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.templating import Jinja2Templates
import os
import uvicorn
from starlette.requests import Request
import neural_network
import pandas as pd
import json

templates = Jinja2Templates(directory="templates")

app = FastAPI(
    openapi_url=f"/api/openapi.json",
    docs_url=f"/api/docs"
)

router = APIRouter()

@router.get('/')
def index(
    request: Request
):
    return templates.TemplateResponse('index_site.html', context = {'request': request})

@router.post('/upload')
def upload(
    file: UploadFile = File(...)
):
    directory = os.path.join(os.getcwd(), 'UPLOAD_FOLDER')

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, file.filename)    
    with open(path, 'wb') as f:
        f.write(file.file.read())
#    json.dump(path, open('path.json', 'w'))

#    # Получение пути к файлу
#    path = json.load(open('path.json', 'r'))
#    path = path.replace('\\\\', '/')
    nn_process = neural_network.neural_network(path)

    # Инициализация датафрейма
    Initial_df = nn_process.Initial_DF_for_process(path)

    if Initial_df is None:
        result = {'result': 'Файл не соответсвует требованиям, должно быть 2 и больше показаний.'}
        print('Файл не соответсвует требованиям, должно быть 2 и более показаний.')
    else:
        # Интерполяция данных
        Consumption_in_day = nn_process.Interpolation_DF(Initial_df)
        # Выделение месячного потребления
        Consumption_in_month = nn_process.Month_consumption(Consumption_in_day, Initial_df)
        #nn_process.Graf_plot(Consumption_in_day, Initial_df, Consumption_in_month)
        # Определение фактора Вид владельца
        Factor_owner_type = nn_process.Factor_of_owner_type(Initial_df)
        # Определение фактора предельного месячного потребления
        Upper_limit = 2000 # Предел месячного потребления
        Factor_UpperLimit = nn_process.Factor_of_month_consumption_UpperLimit(Consumption_in_month, Upper_limit)
        # Определение фактора отрицательного месячного потребления
        Factor_Negative_consumption = nn_process.Factor_of_month_negative_consumption(Consumption_in_month)
        # Определение фактора большого перерыва между датами показаний
        Factor_long_break = nn_process.Factor_of_long_break(Initial_df)
        # Определение фактора большого перерыва между датами показаний с малым значением потребления
        Factor_low_consumption_long_period = nn_process.Factor_of_low_consumption_long_period(Initial_df)

        Factors_consumption = nn_process.Create_factors_consumption(Factor_Negative_consumption,
                                                                     Factor_owner_type,
                                                                     Factor_long_break,
                                                                     Factor_UpperLimit,
                                                                        Factor_low_consumption_long_period)
        # Взаимодействие с моделью нейронной сети
        result = nn_process.neural_network_process(Factors_consumption)
        
        result = {'result': result}
        json.dump(result, open('result.json', 'w'))
        print(result['result'])
    return result['result']

app.include_router(router)


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    

