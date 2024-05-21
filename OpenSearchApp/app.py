import streamlit as st
from PIL import Image
import base64
import yaml
from yaml.loader import SafeLoader
from streamlit_javascript import st_javascript
import streamlit_authenticator as stauth


st.set_page_config(
    
    #page_title="Semantic Search using OpenSearch",
    layout="wide",
    page_icon="/home/ubuntu/images/opensearch_mark_default.png"
)
st.markdown(
    
    """<a href="https://github.com/aws-samples/AI-search-with-amazon-opensearch-service">
    <img src="data:image/png;base64,{}" width="50" height=50", style = 'position: relative;top:2%;left:98%'>
    </a>""".format(
        base64.b64encode(open("/home/ubuntu/AI-search-with-amazon-opensearch-service/OpenSearchApp/images/Github-symbol.png", "rb").read()).decode()
    ),
       
        unsafe_allow_html=True,
    )

# with open('/home/ubuntu/AI-search-with-amazon-opensearch-service/OpenSearchApp/auth.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)
# authenticator = Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )
# name, authentication_status, username = authenticator.login('Login', 'main')


url_ = st_javascript("await fetch('').then(r => window.parent.location.href)")
AI_ICON = "/home/ubuntu/images/opensearch-twitter-card.png"
col_0_1,col_0_2,col_0_3= st.columns([10,50,85])
with col_0_1:
    st.image(AI_ICON, use_column_width='always')
with col_0_2:
    st.header("AI powered OpenSearch")

#st.header(":rewind: Demos available")
st.write("")
#st.write("----")
st.write("Choose a demo")
st.write("")
col_1_1,col_1_2,col_1_3 = st.columns([3,20,80])
with col_1_1:
    st.subheader(":one:")
with col_1_2:
    st.markdown("<p style='fontSize:28px;color:#e28743'>Semantic Search</p>",unsafe_allow_html=True)
with col_1_3:
    demo_1 = st.button(":arrow_forward:",key = "demo_1")
if(demo_1):
    st.switch_page('pages/Semantic_Search.py')
#st.page_link("pages/1_Semantic_Search.py", label=":orange[1. Semantic Search] :arrow_forward:")
#st.button("1. Semantic Search")
image_ = Image.open('/home/ubuntu/images/Semantic_SEarch.png')
new_image = image_.resize((1500, 1000))
new_image.save('images/semantic_search_resize.png')
st.image("images/semantic_search_resize.png")
st.write("")
col_2_1,col_2_2,col_2_3 = st.columns([3,40,65])
with col_2_1:
    st.subheader(":two:")
with col_2_2:
    st.markdown("<p style='fontSize:28px;color:#e28743'>Multimodal Conversational Search</p>",unsafe_allow_html=True)
with col_2_3:
    demo_2 = st.button(":arrow_forward:",key = "demo_2")
if(demo_2):
    st.switch_page('pages/Multimodal_Conversational_Search.py')
#st.header("2. Multimodal Conversational Search")
image_ = Image.open('images/RAG_.png')
new_image = image_.resize((1500, 1000))
new_image.save('images/RAG_resize.png')
st.image("images/RAG_resize.png")

# with st.sidebar:
#     st.subheader("Choose a demo !")




    #  """
    #     <style>
          
    #         [data-testid="stHeader"]::after {
    #             content: "My Company Name";
    #             margin-left: 0px;
    #             margin-top: 0px;
    #             font-size: 30px;
    #             position: relative;
    #             left: 90%;
    #             top: 30%;
    #         }
    #     </style>
    #     """,