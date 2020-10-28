#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import spacy
import pt_core_news_sm 
import string
from nltk import ngrams
from collections import Counter
from operator import itemgetter
from unidecode import unidecode
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import time



# In[2]:

try:
    import streamlit as st
    st.title("Buscador de Vagas PcD")
    st.subheader("Qual vaga você procura?")
    search = st.text_input("Digite aqui o cargo ideal para você:") 
    st.button("BUSCAR")
    search = search.replace(' ','-')
    
    page = requests.get('https://www.vagas.com.br/vagas-de-'+search)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    vagas = soup.find_all(class_ = 'link-detalhes-vaga')
    
    #Armazenando links em uma lista
    lista = []
    for i in vagas:
        lista.append(i.get('href'))
        
    #Testanto hiperlinks da lista
    lista1 = []
    for i in lista:
        lista1.append('https://www.vagas.com.br/'+i)
        
    df_vagas = pd.DataFrame(lista1)
    
    
    # In[3]:
    
    
    df = pd.DataFrame()
    comp = []
    emp = []
    sen = []
    loc = []
    fax_sal = []
    for i in lista1:
        page = requests.get(i)
        soup = BeautifulSoup(page.content, 'html.parser')
        
        competencias = soup.find(class_= 'texto')
        competencias = competencias.get_text(strip = True)
        
        empresa = soup.find('h2', class_ = 'job-shortdescription__company').text.strip()
        
        titulo = soup.find('h1', class_ = 'job-shortdescription__title').text.strip()
        
        senioridade = soup.find('div', {'class': 'job-hierarchylist'}).text.strip().replace('\n', '')
        
        localizacao = soup.find('span', class_ = 'info-localizacao').text.strip()
        
        faixa_salarial = soup.find('div', {'class': 'infoVaga'}).text.strip().replace('\n', '')
        
        comp.append(competencias)
        emp.append(empresa)
        sen.append(senioridade)
        loc.append(localizacao)
        fax_sal.append(faixa_salarial)
        
    df_competencias = pd.DataFrame(comp)
    df_empresa = pd.DataFrame(emp)
    df_senioridade = pd.DataFrame(sen)
    df_localizacao = pd.DataFrame(loc)
    df_faixa_salarial = pd.DataFrame(fax_sal)
    
    df_exp = pd.merge(df_competencias, df_empresa, right_index=True, left_index=True)
    df_exp = pd.merge(df_exp, df_senioridade, right_index=True, left_index=True)
    df_exp = pd.merge(df_exp, df_localizacao, right_index=True, left_index=True)
    df_exp = pd.merge(df_exp, df_faixa_salarial, right_index=True, left_index=True)
    df_exp = pd.merge(df_exp, df_vagas, right_index=True, left_index=True)
    
    df['Competencias'] = comp
    
    
    # In[4]:
    
    
    df['Competencias'] = df['Competencias'].str.replace('\r\n', ' ')
    df['Competencias'] = df['Competencias'].str.replace('• ', ' ')
    df['Competencias'] = df['Competencias'].str.replace('Descrição', ' ')
    df['Competencias'] = df['Competencias'].str.replace(':', ' ')
    df['Competencias'] = df['Competencias'].str.replace('/', ' ')
    df['Competencias'] = df['Competencias'].str.replace('-', ' ')
    
    
    # In[5]:
    
    
    df_exp.columns = ['Competências', 'Empresa', 'Senioridade', 'Localização', 'Faixa Salarial', 'Link vaga']
    
    
    # In[6]:
    
    
    df_exp['Faixa Salarial Inicial'] = df_exp['Faixa Salarial'].str.split("$", n = 2).str[1]
    df_exp['Faixa Salarial Final'] = df_exp['Faixa Salarial'].str.split("$", n = 2).str[2]
    
    
    # In[7]:
    
    
    try:
        df_exp['Faixa Salarial Inicial'] = df_exp['Faixa Salarial Inicial'].str.split(" ").str[1]
        df_exp['Faixa Salarial Final'] = df_exp['Faixa Salarial Final'].str.split(" ").str[1]
    except:
        pass
    
    
    # In[8]:
    
    
    #Removendo acentos e pontuação das colunas 
    for i in range(len(df.Competencias)):
        df.Competencias[i] = unidecode(df.Competencias[i])
        df.Competencias[i] = df.Competencias[i].replace(';',' ')
        df.Competencias[i] = df.Competencias[i].replace('.',' ')
    
    
    # In[9]:
    
    
    #Instanciando NLP
    nlp = spacy.load('pt_core_news_sm')
    
    #Definindo puntuações a serem removidas
    punctuations = string.punctuation
    
    
    # In[10]:
    
    
    #Função que substitui palavras ou termos que são sinonimos
    def junta_sinonimo(x,y):
        df['Competencias'] = df['Competencias'].str.replace(x,y)
    
    
    # In[11]:
    
    
    junta_sinonimo('administracao','administrativas')
    
    
    # In[12]:
    
    
    junta_sinonimo('pacote','office')
    
    
    # In[13]:
    
    
    junta_sinonimo('ensino','medio')
    
    
    # In[14]:
    
    
    deletar_unigram = ['by', 'r','medica','ti','sobre', 'de','e','em','a','com','o','da', 'ou', 'que', 'dos', 'como', 'os', 'ou', 'para', 'do', 'na', 'and',                  'no', 'uma', 'as', 'por', 'mais', 'um', 'areas', 'no', 'conhecimento', 'nos', 'voce', 'para', 'analisar','que','learning','do',                   'machine','que','experiencia','data','modelos','analise','ferramentas','desenvolvimento','etc','tecnicas','programacao','conhecimentos','formacao','modelagem',                  'matematica','big','linguagens','solucoes','computacao','science','banco','area','projetos','negocio','algoritmos','das','to','sistemas','desenvolver',                  'problemas','desejavel','nao','engenharia','processos','superior','analises','ciencia','requisitos','insights','clientes','uso','bancos','ter',                  'empresa','criacao','negocios','avancado','of','capacidade','utilizando','the','ser','se','sao','trabalhar' ,'linguagem','outros','servicos','inteligencia',                  'in','resultados','estruturados','time','power', 'oportunidades','atividades','trabalho','principais','identificar','tempo' 'rj',
                      'brasil', 's', 'f1', 'causal', 'c', 'w', 'remuneracao', 'saiba', 'b', 'descricaoatividades', 'necessario', 'necessarios', 'completo', 'descricaoresponsabilidades',
                      'boa', '17h48','somosumso!e', 'ai', 'gente?quais', 'responsabilidadesorganizar', 'aa', '6x1', 'ja']
    
    
    # In[15]:
    
    
    stopwords = ['de' ,'a' ,'o' ,'que' ,'e' ,'do' ,'da' ,'em' ,'um' ,'para' ,'é' ,'com' ,'não' ,'uma' ,'os' ,
                 'no' ,'se' ,'na' ,'por' ,'mais' ,'as' ,'dos' ,'como' ,'mas' ,'foi' ,'ao' ,'ele' ,'das' ,'tem' ,
                 'à' ,'seu' ,'sua' ,'ou' ,'ser' ,'quando' ,'muito' ,'há' ,'nos' ,'já' ,'está' ,'eu' ,'também' ,
                 'só' ,'pelo' ,'pela' ,'até' ,'isso' ,'ela' ,'entre' ,'era' ,'depois' ,'sem' ,'mesmo' ,'aos' ,
                 'ter' ,'seus' ,'quem' ,'nas' ,'me' ,'esse' ,'eles' ,'estão' ,'você' ,'tinha' ,'foram' ,'essa' ,
                 'num' ,'nem' ,'suas' ,'meu' ,'às' ,'minha' ,'têm' ,'numa' ,'pelos' ,'elas' ,'havia' ,'seja' ,
                 'qual' ,'será' ,'nós' ,'tenho' ,'lhe' ,'deles' ,'essas' ,'esses' ,'pelas' ,'este' ,'fosse' ,
                 'dele' ,'tu' ,'te' ,'vocês' ,'vos' ,'lhes' ,'meus' ,'minhas','teu' ,'tua','teus','tuas','nosso',
                 'nossa','nossos','nossas','dela' ,'delas' ,'esta' ,'estes' ,'estas' ,'aquele' ,'aquela' ,'aqueles',
                 'aquelas' ,'isto' ,'aquilo' ,'estou','está','estamos','estão','estive','esteve','estivemos',
                 'estiveram', 'estava','estávamos','estavam','estivera','estivéramos','esteja','estejamos',
                 'estejam','estivesse', 'estivéssemos','estivessem','estiver','estivermos','estiverem','hei','há',
                 'havemos','hão','houve','houvemos','houveram','houvera','houvéramos','haja','hajamos','hajam',
                 'houvesse','houvéssemos','houvessem', 'houver','houvermos','houverem','houverei','houverá',
                 'houveremos','houverão','houveria','houveríamos', 'houveriam','sou','somos','são','era','éramos',
                 'eram','fui','foi','fomos','foram','fora','fôramos','seja', 'sejamos','sejam','fosse','fôssemos',
                 'fossem','for','formos','forem','serei','será','seremos','serão','seria', 'seríamos','seriam',
                 'tenho','tem','temos','tém','tinha','tínhamos','tinham','tive','teve','tivemos','tiveram',
                 'tivera','tivéramos','tenha','tenhamos','tenham','tivesse','tivéssemos','tivessem','tiver',
                 'tivermos','tiverem','terei','terá','teremos','terão','teria','teríamos','teriam']
    
    
    # In[16]:
    
    
    #Tokenização
    def preprocess(text):
        
        #Instanciando NLP
        doc = nlp(text)
        
        #Selecionando somente Adjetivos e nomes próprios
        tokens = [token for token in doc if token.pos_ == 'ADJ' or token.pos_ == 'PROPN']
        
        #Conversão de Tokens para string
        tokens = [token.orth_ for token in tokens]
           
        #To lower
        tokens = [(token.lower()) for token in tokens]
        
        #Removendo itens da lista de removação manual
        tokens = [token for token in tokens if (token not in deletar_unigram)]
        
        #Removendo stopwords e punctuations
        tokens = [token for token in tokens if (token not in stopwords and token not in punctuations)]  
        
        return ' '.join(tokens)
    
    df['Competencias'] = df['Competencias'].apply(preprocess)
    
    
    # In[18]:
    
    
    df.insert(1, "Curriculo", """pcd ensino responsavel excel analitico tecnologia powerbi pne intelectual
                                 profissional objetivo analista deficiencia software python autismo fisica
                                 visual auditiva""")
    
    
    # In[19]:
    
    
    df
    
    
    # In[20]:
    
    
    from strsimpy.cosine import Cosine
    cosine = Cosine(2)
    df["p0"] = df["Competencias"].apply(lambda s: cosine.get_profile(s)) 
    df["p1"] = df["Curriculo"].apply(lambda s: cosine.get_profile(s)) 
    df["cosine_sim"] = [cosine.similarity_profiles(p0,p1) for p0,p1 in zip(df["p0"],df["p1"])]
    
    
    # In[21]:
    
    
    df = df.drop(["p0", "p1"], axis=1)
    df_exp = pd.merge(df_exp, df, right_index=True, left_index=True)
    df_exp = df_exp.rename(columns={'cosine_sim': 'Nota da vaga'})
    df_exp = df_exp.sort_values(by='Nota da vaga', ascending=False)
    df_exp = df_exp.drop(["Competencias", "Curriculo", "Faixa Salarial"], axis=1)
    df_exp.reset_index(inplace=True)
    df_exp.drop(columns='index', inplace=True)
    	
    time.sleep(200)
except:
    pass

# In[ ]:


st.markdown(df_exp.index.tolist())

