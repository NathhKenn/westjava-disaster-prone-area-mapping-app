import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Konfigurasi halaman
st.set_page_config(page_title="Clustering Wilayah Rawan Bencana", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #7f7f7f;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
        color: #262730;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.markdown('<p class="main-header">🗺️ Analisis Clustering Wilayah Rawan Bencana di Jawa Barat</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Perbandingan Algoritma K-Means dan DBSCAN</p>', unsafe_allow_html=True)

# Fungsi untuk memuat data CSV
@st.cache_data
def load_data(start_year, end_year):
    """
    Memuat dataset bencana dari file CSV dan filter berdasarkan tahun
    Input: start_year dan end_year untuk filter
    Output: DataFrame dengan data bencana yang sudah diagregasi dan dibersihkan
    """
    try:
        df = pd.read_csv('gabungan_data_bencana_per_tahun.csv')
        
        # Filter berdasarkan tahun
        df_filtered = df[(df['tahun'] >= start_year) & (df['tahun'] <= end_year)].copy()
        
        # Agregasi data berdasarkan kabupaten/kota (jumlah kejadian bencana)
        df_agg = df_filtered.groupby(['kode_kabupaten_kota', 'nama_kabupaten_kota']).agg({
            'Banjir': 'sum',
            'Tanah Longsor': 'sum',
            'Cuaca Ekstrem': 'sum',
            'Gempa Bumi': 'sum',
            'Kebakaran': 'sum'
        }).reset_index()
        
        # Rename kolom agar konsisten dengan kode sebelumnya
        df_agg.columns = ['kode_kabupaten_kota', 'nama_kabupaten_kota', 'banjir', 
                          'tanah_longsor', 'cuaca_ekstrem', 'gempa_bumi', 'kebakaran']
        
        # Bersihkan nama kabupaten (hapus prefix "KABUPATEN")
        df_agg['nama_kabupaten_kota_clean'] = df_agg['nama_kabupaten_kota'].apply(
            lambda x: x.replace('KABUPATEN ', '') if 'KABUPATEN' in x else x
        )
        
        return df_agg
    except FileNotFoundError:
        st.error(f"❌ File 'gabungan_data_bencana_per_tahun.csv' tidak ditemukan!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error membaca CSV: {str(e)}")
        st.stop()

# Fungsi untuk memuat file GeoJSON peta Jawa Barat
@st.cache_data
def load_geojson():
    """
    Memuat file GeoJSON untuk visualisasi peta choropleth
    Output: data GeoJSON dan status keberhasilan loading
    """
    geojson_file = 'Jabar_By_Kab.geojson'
    try:
        with open(geojson_file, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        return geojson_data, True
    except:
        return None, False

geojson_data, use_geojson = load_geojson()
geojson_id_field = "KABKOT"

# Sidebar untuk parameter
st.sidebar.header("📅 Pilihan Rentang Tahun Data")

# Slider untuk memilih rentang tahun
year_range = st.sidebar.slider(
    "Pilih Rentang Tahun:",
    min_value=2012,
    max_value=2024,
    value=(2015, 2024),
    step=1,
    help="Pilih rentang tahun data bencana yang akan dianalisis"
)

start_year, end_year = year_range
st.sidebar.info(f"📊 Periode: **{start_year} - {end_year}** ({end_year - start_year + 1} tahun)")

df = load_data(start_year, end_year)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parameter Clustering")

# Parameter K-Means
st.sidebar.subheader("🔵 K-Means")
k_value = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=4, step=1)

st.sidebar.markdown("---")

# Parameter DBSCAN
st.sidebar.subheader("🟣 DBSCAN")
eps_value = st.sidebar.slider("Epsilon (eps)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
min_pts_value = st.sidebar.slider("Min Points", min_value=2, max_value=10, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Total Wilayah: **{len(df)}**")

# Preprocessing data: ekstrak fitur dan normalisasi
features = ['banjir', 'tanah_longsor', 'gempa_bumi', 'cuaca_ekstrem', 'kebakaran']
X = df[features].values

# Standardisasi data agar setiap fitur memiliki skalaza yang sama (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA untuk reduksi dimensi ke 2D untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fungsi untuk melakukan clustering dengan K-Means dan DBSCAN
def perform_clustering(X_scaled, k, eps, min_pts):
    """
    Melakukan clustering dengan K-Means dan DBSCAN, serta menghitung metrik evaluasi
    Input: data yang sudah dinormalisasi, parameter k, eps, dan min_pts
    Output: label cluster dan metrik evaluasi (silhouette score dan DBI)
    """
    # K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_pts)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Hitung metrik evaluasi untuk K-Means
    try:
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
        kmeans_silhouette_samples = silhouette_samples(X_scaled, kmeans_labels)
        kmeans_dbi = davies_bouldin_score(X_scaled, kmeans_labels)
    except:
        kmeans_silhouette = None
        kmeans_silhouette_samples = None
        kmeans_dbi = None
    
    # Hitung metrik evaluasi untuk DBSCAN (tanpa noise points)
    try:
        mask = dbscan_labels != -1
        if mask.sum() > 0 and len(set(dbscan_labels[mask])) > 1:
            dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
            # Buat array silhouette samples dengan -1 untuk noise
            dbscan_silhouette_samples = np.full(len(dbscan_labels), -1.0)
            dbscan_silhouette_samples[mask] = silhouette_samples(X_scaled[mask], dbscan_labels[mask])
            dbscan_dbi = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
        else:
            dbscan_silhouette = None
            dbscan_silhouette_samples = None
            dbscan_dbi = None
    except:
        dbscan_silhouette = None
        dbscan_silhouette_samples = None
        dbscan_dbi = None
    
    return (kmeans_labels, dbscan_labels, kmeans_silhouette, dbscan_silhouette, 
            kmeans_silhouette_samples, dbscan_silhouette_samples, kmeans_dbi, dbscan_dbi)

# Lakukan clustering
(kmeans_labels, dbscan_labels, kmeans_sil, dbscan_sil,
 kmeans_sil_samples, dbscan_sil_samples, kmeans_dbi, dbscan_dbi) = perform_clustering(X_scaled, k_value, eps_value, min_pts_value)

# Tambahkan hasil clustering ke dataframe
df_result = df.copy()
df_result['KMeans_Cluster'] = kmeans_labels
df_result['DBSCAN_Cluster'] = dbscan_labels
df_result['PCA_1'] = X_pca[:, 0]
df_result['PCA_2'] = X_pca[:, 1]

if kmeans_sil_samples is not None:
    df_result['KMeans_Silhouette'] = kmeans_sil_samples
if dbscan_sil_samples is not None:
    df_result['DBSCAN_Silhouette'] = dbscan_sil_samples

# Koordinat geografis setiap kabupaten/kota untuk visualisasi peta
@st.cache_data
def get_coordinates():
    """
    Mendefinisikan koordinat lat/lon untuk setiap kabupaten/kota di Jawa Barat
    Digunakan sebagai fallback jika GeoJSON tidak tersedia
    """
    coords = {
        'BANDUNG': [-7.0051, 107.5662],
        'BANDUNG BARAT': [-6.8622, 107.4917],
        'BEKASI': [-6.2349, 107.1489],
        'BOGOR': [-6.5950, 106.7969],
        'CIAMIS': [-7.3257, 108.3534],
        'CIANJUR': [-6.8167, 107.1392],
        'CIREBON': [-6.7363, 108.5570],
        'GARUT': [-7.2219, 107.9037],
        'INDRAMAYU': [-6.3264, 108.3200],
        'KARAWANG': [-6.3063, 107.3019],
        'KUNINGAN': [-6.9759, 108.4831],
        'MAJALENGKA': [-6.8396, 108.2277],
        'PANGANDARAN': [-7.6859, 108.6500],
        'PURWAKARTA': [-6.5569, 107.4433],
        'SUBANG': [-6.5697, 107.7633],
        'SUKABUMI': [-6.9278, 106.9278],
        'SUMEDANG': [-6.8387, 107.9214],
        'TASIKMALAYA': [-7.3506, 108.2170],
        'KOTA BANDUNG': [-6.9175, 107.6191],
        'KOTA BANJAR': [-7.3705, 108.5389],
        'KOTA BEKASI': [-6.2383, 106.9756],
        'KOTA BOGOR': [-6.5950, 106.8169],
        'KOTA CIMAHI': [-6.8722, 107.5422],
        'KOTA CIREBON': [-6.7063, 108.5570],
        'KOTA DEPOK': [-6.4025, 106.7942],
        'KOTA SUKABUMI': [-6.9278, 106.9278],
        'KOTA TASIKMALAYA': [-7.3257, 108.2170]
    }
    return coords

coordinates = get_coordinates()
df_result['lat'] = df_result['nama_kabupaten_kota_clean'].map(lambda x: coordinates.get(x, [0, 0])[0])
df_result['lon'] = df_result['nama_kabupaten_kota_clean'].map(lambda x: coordinates.get(x, [0, 0])[1])

# Tampilkan metrik evaluasi clustering
st.markdown("### 📊 Evaluasi Performa Clustering")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔵 K-Means Clustering")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        if kmeans_sil is not None:
            st.metric("Silhouette Score", f"{kmeans_sil:.4f}", help="Nilai lebih tinggi = lebih baik (range: -1 to 1)")
        else:
            st.metric("Silhouette Score", "N/A")
    
    with metric_col2:
        if kmeans_dbi is not None:
            st.metric("Davies-Bouldin Index", f"{kmeans_dbi:.4f}", help="Nilai lebih rendah = lebih baik")
        else:
            st.metric("Davies-Bouldin Index", "N/A")
    
    with metric_col3:
        st.metric("Jumlah Cluster", k_value)
    
with col2:
    st.markdown("#### 🟣 DBSCAN Clustering")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        if dbscan_sil is not None:
            st.metric("Silhouette Score", f"{dbscan_sil:.4f}", help="Nilai lebih tinggi = lebih baik (range: -1 to 1)")
        else:
            st.metric("Silhouette Score", "N/A")
    
    with metric_col2:
        if dbscan_dbi is not None:
            st.metric("Davies-Bouldin Index", f"{dbscan_dbi:.4f}", help="Nilai lebih rendah = lebih baik")
        else:
            st.metric("Davies-Bouldin Index", "N/A")
    
    with metric_col3:
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        st.metric("Cluster | Noise", f"{n_clusters} | {n_noise}")

# Elbow Method untuk K-Means
st.markdown("---")
st.markdown("### 📐 Elbow Method - Penentuan K Optimal untuk K-Means")

@st.cache_data
def calculate_elbow_method(X_scaled, k_range=(2, 11)):
    """
    Menghitung Inertia, Silhouette Score, dan DBI untuk berbagai nilai K
    Untuk analisis Elbow Method
    """
    inertias = []
    silhouettes = []
    dbis = []
    k_values = list(range(k_range[0], k_range[1]))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        
        try:
            sil = silhouette_score(X_scaled, labels)
            silhouettes.append(sil)
        except:
            silhouettes.append(0)
        
        try:
            dbi = davies_bouldin_score(X_scaled, labels)
            dbis.append(dbi)
        except:
            dbis.append(0)
    
    return k_values, inertias, silhouettes, dbis

k_values, inertias, silhouettes, dbis = calculate_elbow_method(X_scaled)

# Buat visualisasi Elbow Method (hanya Inertia) dengan detail lengkap
fig_elbow = go.Figure()

# Plot garis dan marker
fig_elbow.add_trace(
    go.Scatter(
        x=k_values,
        y=inertias,
        mode='lines+markers+text',
        name='WCSS (Inertia)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=12, color='#1f77b4', line=dict(width=2, color='white')),
        text=[f'{inertia:.2f}' for inertia in inertias],
        textposition='top center',
        textfont=dict(size=11, color='#1f77b4', family='Arial Black'),
        hovertemplate='<b>K=%{x}</b><br>Inertia (WCSS)=%{y:.4f}<extra></extra>'
    )
)

# Highlight K yang dipilih user dengan marker bintang merah
fig_elbow.add_trace(
    go.Scatter(
        x=[k_value],
        y=[inertias[k_value-2]],
        mode='markers+text',
        name=f'K Terpilih',
        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='darkred')),
        text=[f'K={k_value}<br>{inertias[k_value-2]:.2f}'],
        textposition='bottom center',
        textfont=dict(size=12, color='red', family='Arial Black'),
        hovertemplate=f'<b>K TERPILIH = {k_value}</b><br>Inertia = {inertias[k_value-2]:.4f}<extra></extra>',
        showlegend=True
    )
)

# Update layout dengan grid dan styling
fig_elbow.update_layout(
    title=dict(
        text=f'<b>Elbow Method - Periode {start_year}-{end_year}</b>',
        x=0.5,
        xanchor='center',
        font=dict(size=18, color='#1f77b4')
    ),
    xaxis=dict(
        title=dict(
            text='<b>Jumlah Cluster (K)</b>',
            font=dict(size=14)
        ),
        tickmode='linear',
        tick0=2,
        dtick=1,  # Tampilkan semua nilai K (1, 2, 3, 4, ...)
        gridcolor='lightgray',
        showgrid=True,
        zeroline=False
    ),
    yaxis=dict(
        title=dict(
            text='<b>WCSS (Within-Cluster Sum of Squares) / Inertia</b>',
            font=dict(size=14)
        ),
        gridcolor='lightgray',
        showgrid=True,
        zeroline=False
    ),
    plot_bgcolor='rgba(250,250,250,0.9)',
    height=500,
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        x=0.98,
        y=0.98,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray',
        borderwidth=1
    )
)

st.plotly_chart(fig_elbow, use_container_width=True)

# Informasi interpretasi
st.info("🔹 **Cara Membaca Elbow Method**: Cari titik 'siku' (elbow) pada grafik di mana penurunan inertia mulai melambat secara signifikan. Titik tersebut menunjukkan K optimal di mana penambahan cluster tidak lagi memberikan peningkatan kualitas clustering yang berarti.")

# Tabel ringkasan untuk membantu pemilihan K
st.markdown("#### 📊 Tabel Ringkasan Metrik per K")

elbow_df = pd.DataFrame({
    'K': k_values,
    'Inertia': inertias,
    'Silhouette Score': silhouettes,
    'DBI': dbis
})

# Tambahkan ranking
elbow_df['Rank Silhouette'] = elbow_df['Silhouette Score'].rank(ascending=False).astype(int)
elbow_df['Rank DBI'] = elbow_df['DBI'].rank(ascending=True).astype(int)
elbow_df['Rank Total'] = elbow_df['Rank Silhouette'] + elbow_df['Rank DBI']

# Sort by Rank Total (lower is better)
elbow_df = elbow_df.sort_values('Rank Total')

# Highlight current K
def highlight_current_k(row):
    if row['K'] == k_value:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)

st.dataframe(
    elbow_df.style.apply(highlight_current_k, axis=1)
                  .background_gradient(subset=['Silhouette Score'], cmap='RdYlGn')
                  .background_gradient(subset=['DBI'], cmap='RdYlGn_r')
                  .format({
                      'Inertia': '{:.2f}',
                      'Silhouette Score': '{:.4f}',
                      'DBI': '{:.4f}'
                  }),
    use_container_width=True,
    height=400
)

# Rekomendasi K optimal berdasarkan metrik
best_k_silhouette = elbow_df.loc[elbow_df['Silhouette Score'].idxmax(), 'K']
best_k_dbi = elbow_df.loc[elbow_df['DBI'].idxmin(), 'K']
best_k_combined = elbow_df.iloc[0]['K']

col_rec1, col_rec2, col_rec3 = st.columns(3)

with col_rec1:
    st.metric(
        "🏆 K Terbaik (Silhouette)",
        int(best_k_silhouette),
        help="K dengan Silhouette Score tertinggi"
    )

with col_rec2:
    st.metric(
        "🏆 K Terbaik (DBI)",
        int(best_k_dbi),
        help="K dengan DBI terendah"
    )

with col_rec3:
    st.metric(
        "🏆 K Terbaik (Gabungan)",
        int(best_k_combined),
        help="K dengan ranking gabungan terbaik"
    )

if k_value != best_k_combined:
    st.info(f"💡 **Saran**: Berdasarkan analisis metrik gabungan, K={int(best_k_combined)} mungkin memberikan hasil clustering yang lebih optimal dibanding K={k_value} yang saat ini dipilih.")

# Visualisasi perbandingan metrik evaluasi
st.markdown("---")
st.markdown("### 📈 Perbandingan Metrik Evaluasi (K-Means vs DBSCAN)")

col_metric1, col_metric2 = st.columns(2)

with col_metric1:
    fig_sil = go.Figure()
    
    metrics_data = {
        'Algoritma': ['K-Means', 'DBSCAN'],
        'Silhouette Score': [kmeans_sil if kmeans_sil else 0, dbscan_sil if dbscan_sil else 0]
    }
    
    fig_sil.add_trace(
        go.Bar(
            x=metrics_data['Algoritma'],
            y=metrics_data['Silhouette Score'],
            marker_color=['#1f77b4', '#9467bd'],
            text=[f"{val:.4f}" if val > 0 else "N/A" for val in metrics_data['Silhouette Score']],
            textposition='auto',
            name='Silhouette Score'
        )
    )
    
    fig_sil.update_layout(
        title='Silhouette Score (↑ Lebih Baik)',
        xaxis_title='Algoritma',
        yaxis_title='Silhouette Score',
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig_sil, use_container_width=True)

with col_metric2:
    fig_dbi = go.Figure()
    
    dbi_data = {
        'Algoritma': ['K-Means', 'DBSCAN'],
        'DBI': [kmeans_dbi if kmeans_dbi else 0, dbscan_dbi if dbscan_dbi else 0]
    }
    
    fig_dbi.add_trace(
        go.Bar(
            x=dbi_data['Algoritma'],
            y=dbi_data['DBI'],
            marker_color=['#1f77b4', '#9467bd'],
            text=[f"{val:.4f}" if val > 0 else "N/A" for val in dbi_data['DBI']],
            textposition='auto',
            name='Davies-Bouldin Index'
        )
    )
    
    fig_dbi.update_layout(
        title='Davies-Bouldin Index (↓ Lebih Baik)',
        xaxis_title='Algoritma',
        yaxis_title='DBI',
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig_dbi, use_container_width=True)

# Visualisasi Silhouette Plot
st.markdown("---")
st.markdown("### 📊 Silhouette Analysis")

tab_sil1, tab_sil2 = st.tabs(["🔵 K-Means", "🟣 DBSCAN"])

def create_silhouette_plot(labels, silhouette_samples_data, silhouette_avg, title):
    """
    Membuat visualisasi Silhouette Plot
    """
    if silhouette_samples_data is None or silhouette_avg is None:
        return None
    
    fig = go.Figure()
    
    y_lower = 10
    unique_labels = sorted(set(labels))
    
    for i, cluster in enumerate(unique_labels):
        if cluster == -1:  # Skip noise untuk DBSCAN
            continue
            
        cluster_silhouette_values = silhouette_samples_data[labels == cluster]
        cluster_silhouette_values.sort()
        
        size_cluster = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster
        
        color = f'hsl({(i * 360 / len(unique_labels))}, 70%, 50%)'
        
        fig.add_trace(go.Scatter(
            x=cluster_silhouette_values,
            y=np.arange(y_lower, y_upper),
            mode='lines',
            fill='tozerox',
            name=f'Cluster {int(cluster)}',
            line=dict(color=color, width=0.5),
            fillcolor=color,
            hovertemplate='Silhouette Value: %{x:.3f}<br>Sample: %{y}<extra></extra>'
        ))
        
        y_lower = y_upper + 10
    
    # Garis rata-rata silhouette score
    fig.add_vline(
        x=silhouette_avg,
        line=dict(color="red", dash="dash", width=3),
        annotation_text=f"Avg: {silhouette_avg:.3f}",
        annotation_position="top",
        annotation_font=dict(size=14, color='#000000', family='Arial Black')
    )
    
    fig.update_layout(
        title=dict(
            text=title, 
            font=dict(color='#000000', size=16, family='Arial, sans-serif')
        ),
        xaxis_title=dict(
            text='Silhouette Coefficient',
            font=dict(color='#000000', size=14, family='Arial, sans-serif')
        ),
        yaxis_title=dict(
            text='Sample Index',
            font=dict(color='#000000', size=14, family='Arial, sans-serif')
        ),
        showlegend=True,
        height=500,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            gridcolor='#d0d0d0',
            zerolinecolor='#a0a0a0',
            color='#000000',
            tickfont=dict(color='#000000', size=12),
            showgrid=True,
            zeroline=True
        ),
        yaxis=dict(
            gridcolor='#d0d0d0',
            zerolinecolor='#a0a0a0',
            color='#000000',
            tickfont=dict(color='#000000', size=12),
            showgrid=True,
            zeroline=True
        ),
        font=dict(color='#000000', size=12, family='Arial, sans-serif'),
        legend=dict(
            font=dict(color='#000000', size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#000000',
            borderwidth=1
        )
    )
    
    return fig

with tab_sil1:
    if kmeans_sil_samples is not None and kmeans_sil is not None:
        fig_sil_km = create_silhouette_plot(
            kmeans_labels,
            kmeans_sil_samples,
            kmeans_sil,
            f'K-Means Silhouette Plot (K={k_value})'
        )
        if fig_sil_km:
            st.plotly_chart(fig_sil_km, use_container_width=True)
            st.info("💡 **Interpretasi**: Lebar plot menunjukkan jumlah sampel dalam cluster. Nilai positif menunjukkan sampel terklasifikasi dengan baik. Garis merah putus-putus adalah rata-rata silhouette score.")
    else:
        st.warning("⚠️ Data silhouette tidak tersedia untuk K-Means")

with tab_sil2:
    if dbscan_sil_samples is not None and dbscan_sil is not None:
        # Filter noise points
        mask = dbscan_labels != -1
        fig_sil_db = create_silhouette_plot(
            dbscan_labels[mask],
            dbscan_sil_samples[mask],
            dbscan_sil,
            f'DBSCAN Silhouette Plot (eps={eps_value}, min_pts={min_pts_value})'
        )
        if fig_sil_db:
            st.plotly_chart(fig_sil_db, use_container_width=True)
            st.info("💡 **Interpretasi**: Lebar plot menunjukkan jumlah sampel dalam cluster. Nilai positif menunjukkan sampel terklasifikasi dengan baik. Garis merah putus-putus adalah rata-rata silhouette score.")
    else:
        st.warning("⚠️ Data silhouette tidak tersedia untuk DBSCAN")

# Fungsi untuk menghitung ranking parameter terbaik untuk K-Means
@st.cache_data
def calculate_kmeans_ranking(X_scaled, k_range=(2, 11)):
    """
    Menghitung ranking parameter K terbaik untuk K-Means berdasarkan Silhouette Score dan DBI
    """
    results = []
    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        try:
            sil_score = silhouette_score(X_scaled, labels)
            dbi = davies_bouldin_score(X_scaled, labels)
            results.append({'K': k, 'Silhouette Score': sil_score, 'DBI': dbi})
        except:
            pass
    return pd.DataFrame(results).sort_values('Silhouette Score', ascending=False)

# Fungsi untuk menghitung ranking parameter terbaik untuk DBSCAN
@st.cache_data
def calculate_dbscan_ranking(X_scaled):
    """
    Menghitung ranking parameter eps dan min_samples terbaik untuk DBSCAN
    """
    results = []
    eps_range = np.arange(0.1, 2.1, 0.2)
    min_samples_range = range(2, 6)
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            mask = labels != -1
            n_clusters = len(set(labels[mask]))
            
            if n_clusters > 1 and mask.sum() > 0:
                try:
                    sil_score = silhouette_score(X_scaled[mask], labels[mask])
                    dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
                    n_noise = (labels == -1).sum()
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'Silhouette Score': sil_score,
                        'DBI': dbi,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise
                    })
                except:
                    pass
    
    return pd.DataFrame(results).sort_values('Silhouette Score', ascending=False)

# Tampilkan ranking parameter terbaik
st.markdown("---")
st.markdown("### 🏆 Ranking Parameter Terbaik")

col_rank1, col_rank2 = st.columns(2)

with col_rank1:
    st.markdown("#### 🔵 Ranking K-Means")
    kmeans_ranking = calculate_kmeans_ranking(X_scaled)
    kmeans_ranking['Rank'] = range(1, len(kmeans_ranking) + 1)
    kmeans_ranking = kmeans_ranking[['Rank', 'K', 'Silhouette Score', 'DBI']]
    st.dataframe(
        kmeans_ranking.style.background_gradient(subset=['Silhouette Score'], cmap='RdYlGn')
                           .background_gradient(subset=['DBI'], cmap='RdYlGn_r'),
        height=300, 
        use_container_width=True
    )
    st.caption("📌 Silhouette Score: ↑ lebih baik | DBI: ↓ lebih baik")

with col_rank2:
    st.markdown("#### 🟣 Ranking DBSCAN")
    dbscan_ranking = calculate_dbscan_ranking(X_scaled)
    dbscan_ranking['Rank'] = range(1, len(dbscan_ranking) + 1)
    dbscan_ranking = dbscan_ranking[['Rank', 'eps', 'min_samples', 'Silhouette Score', 'DBI', 'n_clusters', 'n_noise']]
    st.dataframe(
        dbscan_ranking.head(15).style.background_gradient(subset=['Silhouette Score'], cmap='RdYlGn')
                                     .background_gradient(subset=['DBI'], cmap='RdYlGn_r'),
        height=300, 
        use_container_width=True
    )
    st.caption("📌 Silhouette Score: ↑ lebih baik | DBI: ↓ lebih baik")

# Visualisasi peta clustering
st.markdown("---")
st.markdown("### 🗺️ Visualisasi Peta Clustering")

tab1, tab2 = st.tabs(["🔵 K-Means", "🟣 DBSCAN"])

# Fungsi untuk membuat peta choropleth dengan GeoJSON
def create_choropleth_map(df_result, cluster_col, title):
    """
    Membuat visualisasi peta choropleth menggunakan GeoJSON
    """
    df_plot = df_result.copy()
    df_plot[f'{cluster_col}_str'] = df_plot[cluster_col].astype(str)
    
    fig = px.choropleth_mapbox(
        df_plot,
        geojson=geojson_data,
        locations='nama_kabupaten_kota_clean',
        featureidkey=f"properties.{geojson_id_field}",
        color=cluster_col,
        hover_name='nama_kabupaten_kota',
        hover_data={
            'banjir': True,
            'tanah_longsor': True,
            'gempa_bumi': True,
            'cuaca_ekstrem': True,
            'kebakaran': True,
            cluster_col: True,
            'nama_kabupaten_kota': False,
            'nama_kabupaten_kota_clean': False
        },
        color_continuous_scale='Viridis',
        mapbox_style="carto-positron",
        zoom=7.5,
        center={"lat": -7.0, "lon": 107.6},
        opacity=0.7,
        height=600,
        title=title
    )
    
    unique_clusters = sorted(df_result[cluster_col].unique())
    
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(
                text='Cluster',
                font=dict(color='#000000', size=12)
            ),
            tickmode='array',
            tickvals=unique_clusters,
            ticktext=[str(int(c)) if c >= 0 else 'Noise' for c in unique_clusters],
            tickfont=dict(color='#000000', size=11),
            outlinecolor='#000000',
            outlinewidth=1
        )
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#000000', size=12, family='Arial, sans-serif'),
        title=dict(
            font=dict(color='#000000', size=16, family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

with tab1:
    fig_kmeans = create_choropleth_map(
        df_result, 
        'KMeans_Cluster', 
        f"K-Means Clustering (K={k_value})"
    )
    st.plotly_chart(fig_kmeans, use_container_width=True)

with tab2:
    df_result['DBSCAN_Display'] = df_result['DBSCAN_Cluster'].apply(lambda x: -1 if x == -1 else x)
    
    fig_dbscan = create_choropleth_map(
        df_result, 
        'DBSCAN_Display', 
        f"DBSCAN Clustering (eps={eps_value}, min_pts={min_pts_value})"
    )
    st.plotly_chart(fig_dbscan, use_container_width=True)

# Tabel hasil clustering dengan Silhouette Score per wilayah
st.markdown("---")
st.markdown("### 📋 Detail Hasil Clustering")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔵 K-Means")
    display_kmeans = df_result[['nama_kabupaten_kota', 'KMeans_Cluster']].copy()
    if kmeans_sil_samples is not None:
        display_kmeans['Silhouette Score'] = df_result['KMeans_Silhouette']
    display_kmeans = display_kmeans.sort_values('KMeans_Cluster')
    display_kmeans['KMeans_Cluster'] = display_kmeans['KMeans_Cluster'].astype(int)
    st.dataframe(display_kmeans, height=400, use_container_width=True)

with col2:
    st.markdown("#### 🟣 DBSCAN")
    display_dbscan = df_result[['nama_kabupaten_kota', 'DBSCAN_Cluster']].copy()
    if dbscan_sil_samples is not None:
        display_dbscan['Silhouette Score'] = df_result['DBSCAN_Silhouette']
    display_dbscan = display_dbscan.sort_values('DBSCAN_Cluster')
    display_dbscan['DBSCAN_Cluster'] = display_dbscan['DBSCAN_Cluster'].apply(
        lambda x: 'Noise' if x == -1 else str(int(x))
    )
    st.dataframe(display_dbscan, height=400, use_container_width=True)

# Statistik detail per cluster
st.markdown("---")
st.markdown("### 📈 Statistik Detail per Cluster")

tab_stat1, tab_stat2 = st.tabs(["🔵 K-Means", "🟣 DBSCAN"])

with tab_stat1:
    # Hitung rata-rata silhouette score dan DBI per cluster
    cluster_silhouettes = {}
    if kmeans_sil_samples is not None:
        for cluster in sorted(df_result['KMeans_Cluster'].unique()):
            cluster_mask = df_result['KMeans_Cluster'] == cluster
            cluster_silhouettes[cluster] = df_result[cluster_mask]['KMeans_Silhouette'].mean()
    
    for cluster in sorted(df_result['KMeans_Cluster'].unique()):
        cluster_data = df_result[df_result['KMeans_Cluster'] == cluster]
        cluster_sil = cluster_silhouettes.get(cluster, None)
        
        sil_text = f" - Avg Silhouette: {cluster_sil:.4f}" if cluster_sil is not None else ""
        
        with st.expander(f"🔍 Cluster {int(cluster)} ({len(cluster_data)} wilayah{sil_text})", expanded=False):
            col_stat1, col_stat2 = st.columns([2, 1])
            
            with col_stat1:
                display_cols = ['nama_kabupaten_kota'] + features
                if kmeans_sil_samples is not None:
                    display_cols.append('KMeans_Silhouette')
                st.dataframe(cluster_data[display_cols], use_container_width=True)
            
            with col_stat2:
                st.markdown("**Statistik Rata-rata:**")
                for feature in features:
                    st.metric(
                        feature.replace('_', ' ').title(),
                        f"{cluster_data[feature].mean():.1f}",
                        delta=None
                    )

with tab_stat2:
    # Hitung rata-rata silhouette score per cluster untuk DBSCAN
    cluster_silhouettes = {}
    if dbscan_sil_samples is not None:
        for cluster in sorted(df_result['DBSCAN_Cluster'].unique()):
            if cluster != -1:  # Skip noise
                cluster_mask = df_result['DBSCAN_Cluster'] == cluster
                cluster_silhouettes[cluster] = df_result[cluster_mask]['DBSCAN_Silhouette'].mean()
    
    for cluster in sorted(df_result['DBSCAN_Cluster'].unique()):
        cluster_data = df_result[df_result['DBSCAN_Cluster'] == cluster]
        cluster_name = "Noise" if cluster == -1 else f"Cluster {int(cluster)}"
        cluster_sil = cluster_silhouettes.get(cluster, None)
        
        sil_text = f" - Avg Silhouette: {cluster_sil:.4f}" if cluster_sil is not None else ""
        
        with st.expander(f"🔍 {cluster_name} ({len(cluster_data)} wilayah{sil_text})", expanded=False):
            col_stat1, col_stat2 = st.columns([2, 1])
            
            with col_stat1:
                display_cols = ['nama_kabupaten_kota'] + features
                if dbscan_sil_samples is not None and cluster != -1:
                    display_cols.append('DBSCAN_Silhouette')
                st.dataframe(cluster_data[display_cols], use_container_width=True)
            
            with col_stat2:
                if cluster != -1:
                    st.markdown("**Statistik Rata-rata:**")
                    for feature in features:
                        st.metric(
                            feature.replace('_', ' ').title(),
                            f"{cluster_data[feature].mean():.1f}",
                            delta=None
                        )
                else:
                    st.info("Data noise tidak memiliki karakteristik cluster")

# Visualisasi PCA Scatter Plot
st.markdown("---")
st.markdown("### 🎯 Visualisasi Clustering dalam Ruang 2D (PCA)")

tab_pca1, tab_pca2 = st.tabs(["🔵 K-Means", "🟣 DBSCAN"])

with tab_pca1:
    fig_pca_km = px.scatter(
        df_result,
        x='PCA_1',
        y='PCA_2',
        color='KMeans_Cluster',
        hover_name='nama_kabupaten_kota',
        hover_data={
            'banjir': True,
            'tanah_longsor': True,
            'gempa_bumi': True,
            'cuaca_ekstrem': True,
            'kebakaran': True,
            'PCA_1': False,
            'PCA_2': False,
            'KMeans_Cluster': True
        },
        labels={'PCA_1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                'PCA_2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
        title=f'K-Means Clustering - PCA Projection (Total Variance: {pca.explained_variance_ratio_.sum():.1%})',
        color_continuous_scale='Viridis',
        height=500
    )
    fig_pca_km.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig_pca_km, use_container_width=True)

with tab_pca2:
    df_result['DBSCAN_Display_Label'] = df_result['DBSCAN_Cluster'].apply(
        lambda x: 'Noise' if x == -1 else f'Cluster {x}'
    )
    
    fig_pca_db = px.scatter(
        df_result,
        x='PCA_1',
        y='PCA_2',
        color='DBSCAN_Display_Label',
        hover_name='nama_kabupaten_kota',
        hover_data={
            'banjir': True,
            'tanah_longsor': True,
            'gempa_bumi': True,
            'cuaca_ekstrem': True,
            'kebakaran': True,
            'PCA_1': False,
            'PCA_2': False,
            'DBSCAN_Cluster': True
        },
        labels={'PCA_1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                'PCA_2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
        title=f'DBSCAN Clustering - PCA Projection (Total Variance: {pca.explained_variance_ratio_.sum():.1%})',
        height=500
    )
    fig_pca_db.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig_pca_db, use_container_width=True)

# Visualisasi distribusi bencana per cluster
st.markdown("---")
st.markdown("### 📊 Analisis Distribusi Bencana per Cluster")

tab_viz1, tab_viz2 = st.tabs(["🔵 K-Means", "🟣 DBSCAN"])

with tab_viz1:
    # Agregasi data per cluster untuk K-Means
    kmeans_agg = df_result.groupby('KMeans_Cluster')[features].mean().reset_index()
    
    # Bar chart rata-rata kejadian bencana per cluster
    fig_kmeans_bar = go.Figure()
    for feature in features:
        fig_kmeans_bar.add_trace(go.Bar(
            name=feature.replace('_', ' ').title(),
            x=[f'Cluster {int(c)}' for c in kmeans_agg['KMeans_Cluster']],
            y=kmeans_agg[feature],
        ))
    
    fig_kmeans_bar.update_layout(
        title='Rata-rata Kejadian Bencana per Cluster (K-Means)',
        xaxis_title='Cluster',
        yaxis_title='Rata-rata Kejadian',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_kmeans_bar, use_container_width=True)
    
    st.markdown("#### 📦 Box Plot Distribusi Data per Cluster")
    
    for feature in features:
        fig_box = go.Figure()
        
        for cluster in sorted(df_result['KMeans_Cluster'].unique()):
            cluster_data = df_result[df_result['KMeans_Cluster'] == cluster][feature]
            fig_box.add_trace(go.Box(
                y=cluster_data,
                name=f'Cluster {int(cluster)}',
                boxmean='sd',
                hovertemplate='<b>Cluster %{x}</b><br>Nilai: %{y}<br><extra></extra>'
            ))
        
        fig_box.update_layout(
            title=f'Distribusi {feature.replace("_", " ").title()} per Cluster',
            xaxis_title='Cluster',
            yaxis_title=f'Jumlah Kejadian {feature.replace("_", " ").title()}',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_box, use_container_width=True)

with tab_viz2:
    # Filter out noise untuk DBSCAN
    dbscan_clean = df_result[df_result['DBSCAN_Cluster'] != -1]
    
    if len(dbscan_clean) > 0:
        dbscan_agg = dbscan_clean.groupby('DBSCAN_Cluster')[features].mean().reset_index()
        
        # Bar chart rata-rata kejadian bencana per cluster
        fig_dbscan_bar = go.Figure()
        for feature in features:
            fig_dbscan_bar.add_trace(go.Bar(
                name=feature.replace('_', ' ').title(),
                x=[f'Cluster {int(c)}' for c in dbscan_agg['DBSCAN_Cluster']],
                y=dbscan_agg[feature],
            ))
        
        fig_dbscan_bar.update_layout(
            title='Rata-rata Kejadian Bencana per Cluster (DBSCAN)',
            xaxis_title='Cluster',
            yaxis_title='Rata-rata Kejadian',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_dbscan_bar, use_container_width=True)
        st.markdown("#### 📦 Box Plot Distribusi Data per Cluster")
        
        for feature in features:
            fig_box = go.Figure()
            
            for cluster in sorted(dbscan_clean['DBSCAN_Cluster'].unique()):
                cluster_data = dbscan_clean[dbscan_clean['DBSCAN_Cluster'] == cluster][feature]
                fig_box.add_trace(go.Box(
                    y=cluster_data,
                    name=f'Cluster {int(cluster)}',
                    boxmean='sd',
                    hovertemplate='<b>Cluster %{x}</b><br>Nilai: %{y}<br><extra></extra>'
                ))
            
            fig_box.update_layout(
                title=f'Distribusi {feature.replace("_", " ").title()} per Cluster',
                xaxis_title='Cluster',
                yaxis_title=f'Jumlah Kejadian {feature.replace("_", " ").title()}',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("⚠️ DBSCAN tidak menghasilkan cluster (semua data adalah noise)")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #7f7f7f; padding: 20px;'>
    <p><b>Analisis Clustering Wilayah Rawan Bencana Jawa Barat ({start_year} - {end_year})</b></p>
    <p><b>Metrik Evaluasi:</b></p>
    <p>• Silhouette Score: Nilai lebih tinggi = clustering lebih baik (range: -1 to 1)</p>
    <p>• Davies-Bouldin Index (DBI): Nilai lebih rendah = clustering lebih baik</p>
</div>
""", unsafe_allow_html=True)