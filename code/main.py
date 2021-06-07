import pandas as pd

test=pd.read_csv('./datasets/tabular-playground-series-may-2021/test.csv')
y0=pd.read_csv("./datasets/lgbm/lgbm_optuna_tpe.csv") # Optuna(lgbm)-tpe
y1=pd.read_csv("./datasets/lgbm/lgbm_tuner.csv") # Optuna(lgbm)-tuner
y2=pd.read_csv("./datasets/lgbm/lgbm_blend.csv") # Previous Blending
y3=pd.read_csv("./datasets/mljar/mljar_1.csv")  # mljar
y4=pd.read_csv("./datasets/keras/hydra_df_blended.csv")  # Keras Hydra
y5=pd.read_csv("./datasets/lightautoml/lightautoml_1.csv") # LightAutoML0
y6=pd.read_csv("./datasets/lightautoml/lightautoml_2.csv") # LightAutoML1
y7=pd.read_csv("./datasets/lightautoml/lightautoml_3.csv") # LightAutoML2
y8=pd.read_csv("./datasets/lightautoml/lightautoml_6.csv") # LightAutoML3 (Perfect)
y9=pd.read_csv("./datasets/lightautoml/lightautoml_8.csv") # LightAutoML & Catboost blend
y10=pd.read_csv("./datasets/previous/1.07068.csv") # Previous Blending
y11=pd.read_csv("./datasets/previous/1.08386.csv") # Previous Blending

weight = (1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1, 1.2, 1.2)

def blend(classname, weight):
    weightSum = sum(weight)
    return (y0[classname]*weight[0]+y1[classname]*weight[1]+y2[classname]*weight[2]+y3[classname]*weight[3]+y4[classname]*weight[4]+y5[classname]*weight[5]+y6[classname]*weight[6]+y7[classname]*weight[7]+y8[classname]*weight[8]+y9[classname]*weight[9]+y10[classname]*weight[10]+y11[classname]*weight[11]) / weightSum

sub = pd.DataFrame({
        "id": test.id,
        "Class_1": blend("Class_1", weight),
        "Class_2": blend("Class_2", weight),
        "Class_3": blend("Class_3", weight),
        "Class_4": blend("Class_4", weight)
    })

sub.to_csv('blend_v11_fin.csv', index=False)




