import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Вычисляет долю правильных ответов (Какая доля от всех ответов — правильная)
    """
    # сравниваем векторы [1, 0, 1] == [1, 1, 1] -> [True, False, True]
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    """
    Вычисляет точность (Если модель сказала, что это класс 1, насколько ей можно верить)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred):
    """ 
    Вычисляет полноту (Какую долю реальных объектов 1-го класса мы смогли найти)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    """
    Находим гармоническое среднее между precision и recall, 
    высокий показатель только при высоких значениях этих двук параметров 
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def roc_curve(y_true, y_probs):
    """
    Возвращает fpr, tpr и thresholds для построения ROC-кривой
    """
    # Берем уникальные значения вероятностей как пороги и сортируем по убыванию
    thresholds = np.sort(np.unique(y_probs))[::-1]
    
    # Добавляем порог > 1.0, чтобы кривая гарантированно начиналась из (0,0)
    thresholds = np.insert(thresholds, 0, thresholds[0] + 0.1)
    
    tpr_list = []
    fpr_list = []
    
    P = np.sum(y_true == 1) # Всего реальных единиц
    N = np.sum(y_true == 0) # Всего реальных нулей
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        tpr = tp / P if P > 0 else 0
        fpr = fp / N if N > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return np.array(fpr_list), np.array(tpr_list), thresholds

def roc_auc_score(y_true, y_probs):
    """
    Площадь под ROC-кривой (метод трапеций)
    """
    # Получаем массивы FPR (False Positive Rate) и TPR (True Positive Rate) для всех возможных порогов
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    # Формула площади трапеции: sum( dx * (y1 + y2) / 2 )
    auc = np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2)
    return auc