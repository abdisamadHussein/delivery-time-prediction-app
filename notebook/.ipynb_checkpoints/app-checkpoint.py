import pickle
import streamlit as st
import pandas as pd

# loading the trained model
pickle_in = open('predictsales.pkl', 'rb')
predictsales = pickle.load(pickle_in)

def main():
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Predict the sales</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    # 2. Loading the data

    # following lines create boxes in which user can enter data required to make prediction

    Item_Weight = st.number_input("Item Weight")
    Item_Fat_Content = st.selectbox('Item Fat Content',('Low Fat', 'Regular'))
    Item_Type = st.selectbox('Item Type',('Dairy', 'Drinks', 'Others', 'Fruits'))
    Item_MRP = st.number_input("Item MRP")
    Outlet_Size = st.selectbox('Outlet Size',('Medium', 'High', 'Small'))
    result =""

    if st.button("Check"):
        result = prediction(Item_Weight, Item_Fat_Content, Item_Type, Item_MRP, Outlet_Size)
        st.success('Predicted sales {}'.format(result))

def prediction(Item_Weight, Item_Fat_Content, Item_Type, Item_MRP, Outlet_Size):

    data = pd.DataFrame({'Item_Weight': [Item_Weight], 'Item_Fat_Content': [Item_Fat_Content], 
                         'Item_Type': [Item_Type], 'Item_MRP': [Item_MRP], 'Outlet_Size': [Outlet_Size]})
    num_data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Size']).astype(float)

    # Ensure collected data has all columns present in 'data'
   
    all_cols = ['Item_Weight', 'Item_MRP', 'Item_Fat_Content_Low Fat',
       'Item_Fat_Content_Regular', 'Item_Type_Dairy', 'Item_Type_Drinks',
       'Item_Type_Fruits', 'Item_Type_Others', 'Outlet_Size_High',
       'Outlet_Size_Medium', 'Outlet_Size_Small']
    missing_cols = set(all_cols) - set(num_data.columns)
    for col in missing_cols:
        num_data[col] = 0
    dummies = num_data[all_cols]

    # 3. Building the model to automate Loan Eligibility

    prediction = predictsales.predict(dummies)
    return prediction

if __name__=='__main__':
    main()

