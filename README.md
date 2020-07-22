# 30Music-artist-recommendation

## Dataset
[30Music Dataset](http://recsys.deib.polimi.it/datasets/)

## Repository organisation

* Data preparation: [data_preparation.ipynb](data_preparation.ipynb)

* First model: baseline. Collaborative filtering for implicit feedback datasets: [collaborative_filtering.ipynb](collaborative_filtering.ipynb)

* Second model. Collaborative filtering for implicit feedback datasets enhanced with user likes: [collaborative_filtering_enhanced.ipynb](collaborative_filtering_enhanced.ipynb)

* Third model. item2vec embeddings: [item2vec.ipynb](item2vec.ipynb)

* Comparison of the models: [methods_quality_comparison.ipynb](methods_quality_comparison.ipynb)

* Helper functions used in the notebooks: [utils](utils/)

To run the code, you have to put 30Music dataset in the folder named ThirtyMusic to the project folder. If you already have standard Python3 libraries for data analysis (for example, you have Anaconda installed), you only have to install library [implicit](https://github.com/benfred/implicit) to run the CF notebooks and [gensim](https://pypi.org/project/gensim/) to train item2vec embeddings. 


## Process description
1. Я подготовила данные: перешла от истории прослушиваний треков к истории прослушиваний исполнителей. Также я подготовила данные о лайках пользователей (love.idomaar). Процесс можно посмотреть в ноутбуке 
[data_preparation.ipynb](data_preparation.ipynb).

2. Так как в задании не было указано, какого типа должна быть рекомендательная система - должна она оценивать рейтинги для каждого юзера и исполнителя или рекоммендовать топ-N исполнителей, я решила выбрать второй вариант. 

Я решила разбить данные на Train и Test по пользователям - 80% пользователей я оставила в трейне, остальные оказались в тестовом датасете. 
Я разделила данные таким образом, потому что было бы хорошо построить модель, способную рекомендовать исполнителей новым юзерам, то есть тем, кого не было в трейнинг сете. В качестве сета для валидации я решила использовать hit rate и left-one-out технику. Для 20% юзеров из трейнинг сета я удалила по одному исполнителю, с которым этот юзер взаимодействовал (слушал его треки), и составила из этих пар юзер-исполнитель left-one-out сет. Таким образом, я пыталась рекомендовать топ-20 лучших исполнителей для этих юзеров, и если среди них был исполнитель, которого я отбросила, я считала это за hit. Поделив на количество юзеров, получала hit rate.  Так можно оценить, может ли модель порекомендовать исполнителя, которого даный юзер на самом деле слушал, но которого не было в трейнинг сете, поэтому, я решила, что это хорошая техника для оценки модели.

Я использовала hit rate скор, чтобы выбрать лучшую модель при тьюнинге гиперпараметров, а потом делала предсказания на тест сете, используя модель с лучшими параметрами. Затем, я делала эвалюэйшн уже на тест сете - считала recall и MAPk.


3. В качестве бейзлайн модели я выбрала **коллаборативную фильтрацию для implicit feedback датасетов** (CF). Также, я пыталась улучшить CF модель, используя информацию о лайках, которые пользователь поставил трекам (relations/love.idomaar). В качестве более сложной модели я натренировала **item2vec** эмбеддинги с помощью gensim word2vec.

#### Collaborative filtering for implicit feedback datasets

Я выбрала коллаборативную фильтрацию, так как это один из самых популярных методов для построения рекомендательных систем. Он дает неплохие результаты, и натренировать модель почти не занимает времени. Так как задача была найти векторные представления для исполнителей, такие, чтобы можно было находить похожих исполнителей, я использовала коллаборативную фильтрацию, чтобы извлечь item factors и посмотреть, насколько они применимы для того, чтобы искать похожих исполнителей используя cosine similarity. 

   В данном случае, у нас нет рейтингов исполнителей или еще каких-то оценок, данных пользователями (кроме лайков юзеров, которые я стала использовать позже), то есть это данные типа implicit feedback. Я применяла метод, описанный в статье [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf). Этот метод реализован в библиотеке [implicit](https://github.com/benfred/implicit), которая очень удобна в использовании и имеет хорошую документацию. Чтобы получить векторные представления исполнителей, я составила матрицу user-artist, в которой значением на пересечении строки юзера и столбца исполнителя является количество прослушиваний данного исполнителя юзером. Я применила Alterating Least Squares, чтобы получить разложение матрицы. Таким образом, я получила 2 матрицы - вектора юзеров и исполнителей. 

   Я натренировала две CF модели с разными параметрами (разная размерность итоговых векторов - 50 и 100 соответственно), чтобы для каждой посчитать hit rate и тем самым оценить, какая из них лучше. Hit rate оказался выше для второй модели. Я использовала ее, чтобы сделать предсказания (рекомендации) на тест сете.  

Так как в тест сете у нас находятся юзеры, которых модель не видела, а у коллаборейтив фильтеринга, основанного на SVD, с этим проблемы, я решила использовать усредненный вектор исполнителей, которых слушал юзер из тест сета, чтобы по этому вектору находить наиболее близкие к нему вектора исполнителей и рекомендовать их юзеру. Для этого я использовала cosine similarity. Таким образом, мы рекомендуем новому пользователю исполнителей, которые похожи на набор исполнителей, которых он прослушал до этого.

Для каждого пользователя из тест сета я разделила список прослушанных исполнителей на две равные части. Далее, я брала одну часть списка, брала вектора для каждого исполнителя, и усредняла их. По усредненному вектору искала наиболее похожих исполнителей, но таких, которых не было в той части списка, по которой мы рекомендуем. Я находила топ-20 таких исполнителей, и считала  MAPk (mean average precision at k) и recall, сравнивая предсказания с другой частью списка. Я использовала MAPk и recall, потому что они являются популярными метриками для оценки топ-N рекомендательных систем.

#### Collaborative filtering for implicit feedback datasets enhanced
Я попыталась улучшить описанную выше CF модель, используя данные о лайках. Для этого, я изменила матрицу взаимодействий юзер-исполнитель, где в качестве значений стала использовать не просто количество прослушиваний данного исполнителя юзером, а количество прослушиваний, просуммированное с количеством лайков с весом 20. Я выбрала большой вес, так как лайк - это важный индикатор того, что исполнитель нравится юзеру, поэтому по логике лайк должен иметь гораздо больший вес, чем просто прослушивание. Я повторила те же действия, что и для обычного CF, но используя уже новую матрицу взаимодействий.



#### item2vec embeddings

В качестве третьей модели я решила натренировать [item2vec](https://arxiv.org/vc/arxiv/papers/1603/1603.04259v2.pdf) эмбеддинги, используя word2vec из gensim. Word2vec модели показали себя очень хорошо в NLP, они дают отличные результаты для построения векторных представлений слов. Я решила, что это должно сработать и для того, чтобы искать похожих исполнителей, и использовала последовательности из id исполнителей в порядке их прослушивания для каждого юзера из трейнинг сета вместо предложений в обычном word2vec. Так, исполнители, которые часто встречаются вместе, должны иметь похожие вектора. 

Я натренировала несколько word2vec моделей с разными параметрами и для каждой посчитала hit rate на left-one-out датасете. Я выбрала для проверки на тест сете лучшую по hit rate модель.

Я использовала эти модели для рекомендаций на тест сете, и опять же, посчитала MAPk (MAP@20) и recall. Для рекомендаций я использовала тот же способ, что и для CF (брала усредненный вектор, полученный из векторов прослушанных исполнителей, и использовала cosine similarity, чтобы найти топ-20 похожих).


4. Сравнение моделей и статистическая значимость полученных результатов

Я сравнила средние MAP@20 и recall для CF и item2vec моделей. Также я сравнила распределения MAP@20 и recall оценок, полученные для каждого юзера в тест сете, и hit rates для каждой из моделей. Для сравнения я использовала Mann-Whitney U rank test, так как он позволяет оценить различия между двумя выборками. Оказалось, что по всем показателям, обычный CF все-таки работает лучше на задаче Artist Recommendation.

Также я сравнила топ-20 похожих исполнителей, полученных с помощью этих трех моделей, для трех исполнителей:

* Ariana Grande (CF vs CF enhanced vs item2vec)

![Similar to Ariana Grande: CF vs item2vec](images/1_joined.jpeg)

* Freddie Mercury (CF vs CF enhanced vs item2vec)

![Similar to Freddie Mercury: CF vs item2vec](images/2_joined.jpeg)

* Mac Miller feat. Action Bronson (CF vs CF enhanced vs item2vec)

![Similar to Mac Miller feat. Action Bronson: CF vs item2vec](images/3_joined.jpeg)


Если сравнивать топ-20 исполнителей, полученных с помощью CF и item2vec (human evaluation), то мне кажется, что item2vec находит похожих исполнителей лучше. 










