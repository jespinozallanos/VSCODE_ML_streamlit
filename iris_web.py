import streamlit as st
import pickle
import pandas as pd

 #probar streamlit
 #en terminal:  streamlit hello
 
# extraccion datos
with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)

with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)

with open('svc_m.pkl', 'rb') as sv:
    svc_m = pickle.load(sv)

#funcion para clasificar las plantas: 1 para virginica, 0 para setosa
def classify(num):
    if num == 0:
        return 'setosa'
    elif num == 1:
        return 'versicolor'
    else:
        return 'virginica'

def main():
    #titulo
    st.title('modelamiento de iris por julio')
    
    #titulo de sidebar
    st.sidebar.header('User Input parameters')
    
    #funcion para poner parametros en sidebar
    def user_input_parameters():
        sepal_length = st.sidebar.slider('sepal lenght', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('petal lenght', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
                }
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()
    
    #escoger modelo preferido
    option = ['linear Regression', 'logistic regression', 'SVM']
    model =st.sidebar.selectbox('wich model do you like to use??', option)
    
    st.subheader('user input parameters')
    st.subheader(model)
    st.write(df)
    
    
    if st.button('RUN'):
        if model == 'linear Regression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'logistic regression':
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_m.predict(df)))
    
    
#metodo de entrada
if __name__ == '__main__':
    main()
    
    
# para correr en powershell:
# streamlit run iris_web.py
a=1

