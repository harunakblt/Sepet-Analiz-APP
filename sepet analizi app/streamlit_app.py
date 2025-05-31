import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st

st.set_page_config(page_title="Sepet Analizi", layout="wide")

# Başlık
st.title("🛒 Sepet Analizi - Ürün Öneri Sistemi")

# Veri yükleme
@st.cache_data
def load_data():
    return pd.read_excel("satis_uretim_final.xlsx")

df = load_data()
st.subheader("📊 Ham Veri")
st.dataframe(df.head())

# Sepet formatına çevirme
basket = df.groupby(['siparis_veren', 'urun_adi']).size().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

st.subheader("🧺 Sepet Formatında Veri")
st.dataframe(basket.head())

# Ayar: minimum support & confidence
st.sidebar.header("🔧 Ayarlar")
min_support = st.sidebar.slider("Minimum Support", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.3, step=0.05)

# Apriori analizi
frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules = rules.sort_values("confidence", ascending=False)

# Kuralları göster
st.subheader("📌 İlişki Kuralları")
rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
st.dataframe(rules_display)

# CSV indir
csv = rules_display.to_csv(index=False).encode('utf-8')
st.download_button("📥 Kuralları CSV Olarak İndir", csv, "kurallar.csv", "text/csv")

# Ürün arama ve öneri
st.subheader("🎯 Ürün Bazlı Öneri")

urunler = sorted(df['urun_adi'].unique())
arama = st.text_input("🔍 Ürün ara veya seç:", "")
secili_urun = st.selectbox("Veya ürün listeden seç:", urunler)

# Arama öncelikli
urun = arama if arama in urunler else secili_urun

# Öneri getir
oneriler = rules[rules['antecedents'].apply(lambda x: urun in x)]

st.markdown(f"**'{urun}' ürününü alan müşterilerin alma eğiliminde olduğu diğer ürünler:**")
st.dataframe(oneriler[['consequents', 'support', 'confidence', 'lift']])
