Измерение отношения сигнала к шуму.

На входе программы последовательность изображенией nd2, а так же z-project в формате jpg, полученный в ImageJ. На изображении выделяются пары точка-фон, и по этим координатам рассчитывается интенсивность в прямоугольниках на последовательности. Разница интенсивностей для каждой пары созраняется в массив и изображается на графике от времени.

Последовательность действий:
1. Убедиться, что на вход программе подаются правильные файлы (window.py строка 67 для z-project, main.py строка 8 для nd2 последовательности) 
2. В открывшемся информационном окне нажать OK после прочтения
3. На изображениии, которое отрылось затем, нажать двойным щелчком на светящуюся точку, она обведется прямоугольником, после этого двойным щелчком выделить фон поблизости. Можно выдельить еще сколько угодно пар
4. Нажать Enter для начала обработки. Откроется несколько окон с графиками для каждой точки

Можно воспроизвести последовательность nd2 как видео, для этого можно запустить код interface.py
В  файле requirements.txt лежат все необходимые библиотеки, их можно как-то автоматически подгрузить в новый environment, чтобы были все необходимые версии