import streamlit as st
import pandas as pd
import spacy
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import uuid
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
import tabula
import io
import re
from datetime import datetime

# Configure embedding and LLM
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=60.0)

st.title("IntelliQuery: AI-Powered Data Insights and Automation Bot")

# Initialize session state
for key in ['query','outputs','df','date_col','numeric_cols']:
    if key not in st.session_state:
        st.session_state[key] = [] if key=='outputs' else '' if key=='query' else None

# File upload section
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV, Excel, or PDF file", type=["csv","xlsx","pdf"], key="file_uploader")

# Clear button
if st.button("Clear"):
    for key in ['query','outputs','df','date_col','numeric_cols']:
        st.session_state[key] = [] if key=='outputs' else '' if key=='query' else None
    st.experimental_rerun()

if uploaded_file:
    try:
        # Read dataset
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            dfs = tabula.read_pdf(io.BytesIO(uploaded_file.read()), pages='all', multiple_tables=True)
            df = dfs[0] if dfs else pd.DataFrame()

        # Clean and preprocess
        df = df.dropna(how='all')
        st.session_state.df = df

        # Detect date and numeric columns
        date_col = None
        numeric_cols = []
        parse_failures = []
        for col in df.columns:
            # Date detection
            if any(k in col.lower() for k in ['date','period','time','updated']) and 'year' not in col.lower():
                ts = df[col].astype(str).str.strip()
                parsed = None
                for fmt in ['%Y-%m-%d','%m/%d/%Y','%Y.%m','%Y-%m-%d %H:%M:%S']:
                    pdts = pd.to_datetime(ts, format=fmt, errors='coerce')
                    if pdts.notna().sum()>len(ts)*0.8:
                        df[col] = pdts; date_col=col; break
                if not date_col:
                    # heuristic fallback
                    sample = ts.sample(min(10,len(ts)))
                    if sample.str.match(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}').mean()>0.5:
                        pdts = pd.to_datetime(ts, errors='coerce')
                        if pdts.notna().sum()>len(ts)*0.8:
                            df[col]=pdts; date_col=col
                        else: parse_failures.append(col)
                    else:
                        parse_failures.append(col)
            # Numeric detection
            if pd.api.types.is_numeric_dtype(df[col]) or any(k in col.lower() for k in ['value','sales','profit','cost','count','rating','warranty']):
                numeric_cols.append(col)
            # Cast object to str
            if df[col].dtype=='object':
                df[col] = df[col].astype(str)

        if parse_failures:
            msg = f"Warning: Date parsing failed for columns: {parse_failures}"
            st.warning(msg)
            st.session_state.outputs.append(msg)

        if not numeric_cols:
            st.error("No numeric columns found in the dataset.")
            st.stop()

        # Clean numeric columns: force to float
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # convert object to float
            df[col] = df[col].fillna(0)  # handle NaNs

        st.session_state.date_col = date_col
        st.session_state.numeric_cols = numeric_cols

        # Preview
        st.session_state.outputs.append("Dataset Preview:")
        st.write(st.session_state.outputs[-1]); st.dataframe(df.head())
        st.session_state.outputs.append(f"Shape: {df.shape}"); st.write(st.session_state.outputs[-1])
        st.session_state.outputs.append(f"Date Column: {date_col}"); st.write(st.session_state.outputs[-1])
        st.session_state.outputs.append(f"Numeric Columns: {numeric_cols}"); st.write(st.session_state.outputs[-1])

        # NLP + vector index
        nlp = spacy.load("en_core_web_sm")
        chroma_client = chromadb.Client()
        coll_name = f"dataset_idx_{uuid.uuid4().hex[:8]}"
        coll = chroma_client.get_or_create_collection(coll_name)
        store = ChromaVectorStore(chroma_collection=coll)
        docs = [Document(text=str(r.to_dict())) for _,r in df.iterrows()]
        index = VectorStoreIndex.from_documents(docs, vector_store=store)

        # Query UI
        st.header("Query Dataset")
        query = st.text_input("Enter query:", value=st.session_state.query, key="query_input")
        st.session_state.query = query

        if query:
            doc = nlp(query)
            st.session_state.outputs.append(f"Parsed Entities: {[ (e.text,e.label_) for e in doc.ents ]}")
            st.write(st.session_state.outputs[-1])

            # Determine if plotting
            is_plot = 'plot' in query.lower() or 'chart' in query.lower()

            if is_plot:
                # Let user pick axes
                plot_cols = numeric_cols + ([date_col] if date_col else [])
                x_col = st.selectbox("X axis:", plot_cols, index=plot_cols.index(date_col) if date_col in plot_cols else 0)
                y_col = st.selectbox("Y axis:", numeric_cols, index=0)
                plot_type = st.radio("Plot type:", ("Line","Scatter","Bar"))
                group_choices = [None] + [c for c in df.columns if df[c].nunique()<50]
                group_col = st.selectbox("Color/Group by:", group_choices, index=0)

                # Build fig
                if plot_type=='Scatter':
                    fig = px.scatter(df, x=x_col, y=y_col, color=group_col, title=f"{y_col} vs {x_col}")
                elif plot_type=='Line':
                    fig = px.line(df, x=x_col, y=y_col, color=group_col, title=f"{y_col} over {x_col}")
                else:
                    fig = px.bar(df, x=x_col, y=y_col, color=group_col, title=f"{y_col} by {x_col}")

                # Ensure axes autorange
                fig.update_xaxes(autorange=True)
                fig.update_yaxes(autorange=True)

                st.plotly_chart(fig, use_container_width=True)
                st.session_state.outputs.append(f"Plotted {y_col} vs {x_col} ({plot_type})")
                st.write(st.session_state.outputs[-1])

            else:
                # Fallback to aggregation or LLM
                ops = {'average':'mean','sum':'sum','max':'max','median':'median','correlation':'correlation'}
                op = next((v for k,v in ops.items() if k in query.lower()), None)
                if op=='correlation' and len(numeric_cols)>1:
                    c1, c2 = numeric_cols[0], numeric_cols[1]
                    df2 = df[[c1,c2]].dropna().astype(float)
                    if len(df2)>1:
                        corr, _ = pearsonr(df2[c1], df2[c2])
                        st.write(f"Correlation {c1}-{c2}: {corr:.2f}")
                        cm = df[numeric_cols].corr()
                        hm = go.Figure(data=go.Heatmap(z=cm.values, x=cm.columns, y=cm.index))
                        st.plotly_chart(hm, use_container_width=True)
                elif op and any(col for col in numeric_cols):
                    grp = next((c for c in df.columns if df[c].nunique()>1), None)
                    if grp:
                        result = df.groupby(grp)[numeric_cols[0]].agg(op)
                        st.write(f"{op.title()} by {grp}:")
                        st.dataframe(result)
                    else:
                        val = getattr(df[numeric_cols[0]], op)()
                        st.write(f"{op.title()} of {numeric_cols[0]}: {val:.2f}")
                else:
                    qe = index.as_query_engine()
                    resp = qe.query(query)
                    st.write(f"LLM Response: {resp}")

    except Exception as e:
        st.error(f"Error: {e}")

if __name__=='__main__':
    st.write("App is running...")
