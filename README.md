Код, расположенный в этом репозитории, имеет мало шансов вновь запуститься и отработать - в момент написания не было уделено достаточно внимания ни версионированию кода, ни документированию, ни сохранению адекватного requirements.txt.

Работа велась либо в Spyder IDE, либо в окнах на PyQt.

Предполагаемый пайплайн работы следующий: 

1. Конвертировать .bin файлы из PowerGraph с помощью  в numpy-массивы [bin_reader_5ch_act_1.py](bin_reader_5ch_act_1.py)
2. Создать и сохранить вейвлет-преобразование полученных данные [SWT_GENERATION.py](SWT_GENERATION.py)
3. Выделить всплески в спектрограмме
4. Отсмотреть моменты всплесков в GUI, сохраняя начало и конец похожих на искомый феномен, предположительно - [event_checker1.py](event_checker1.py). Результаты сохраняются в формате .csv
5. Рассмотреть полученные распределения, предположительно [distr_processing.py](distr_processing.py)
