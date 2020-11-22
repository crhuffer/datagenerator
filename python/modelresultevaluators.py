import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluatemodelresultsv1(y_actual, y_predict, suffix = ''):

    # returning the results as a dict to make it easy to combine with other results using dict1.update(dict2)
    dict_results = dict()
    dict_results['rmse{}'.format(suffix)] = mean_squared_error(y_actual, y_predict)**0.5
    dict_results['mae{}'.format(suffix)] = mean_absolute_error(y_actual, y_predict)
    dict_results['r2_score{}'.format(suffix)] = r2_score(y_actual, y_predict)

    # dict_results['rmse{}'.format(suffix)] = [mean_squared_error(y_actual, y_predict) ** 0.5]
    # dict_results['mae{}'.format(suffix)] = [mean_absolute_error(y_actual, y_predict)]
    # dict_results['r2_score{}'.format(suffix)] = [r2_score(y_actual, y_predict)]

    return dict_results


if __name__ == '__main__':
    y_actual = [0.0, 1.0, 2.0]
    y_predict = [0.1, 1.1, 2.2]
    df_results = pd.DataFrame(evaluatemodelresultsv1(y_actual, y_predict), index=[0])

    # Test that a dict format will make it easy to append results later.
    print(df_results)
    print()

    df_results_temp = df_results.copy()
    print(df_results_temp)
    print()

    df_results = pd.concat([df_results, df_results_temp], ignore_index=True, axis=0)
    print(df_results)

    df_results_test = pd.DataFrame({'rmse': {0: 0.14142135623730961, 1: 0.14142135623730961}, 'mae': {0: 0.13333333333333341, 1: 0.13333333333333341}, 'r2_score': {0: 0.97, 1: 0.97}})

    assert df_results.equals(df_results_test)

# %%

