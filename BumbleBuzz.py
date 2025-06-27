import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import time

st.set_page_config(
    page_title="BumbleBuzz", 
    layout="wide",
    initial_sidebar_state="expanded"
)

#CSS Part 
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .stats-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .audio-info {
        background: #030303;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .decision-buttons {
        margin: 1rem 0;
    }
    .nav-buttons {
        margin-top: 1rem;
    }
    .stButton > button {
        width: 100%;
        margin: 0.2rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

#top heading
st.markdown("""
    <div class="main-header">
        <h1>🐝 Bumble Buzz 🐝</h1>
        <p>Manual verifications</p>
    </div>
""", unsafe_allow_html=True)

#side bar 
with st.sidebar:
    st.header("📁 File Selection")
    
    csv_file = st.file_uploader(
        "Choose CSV file", 
        type="csv",
        help="Upload the CSV file containing audio metadata"
    )
    
    audio_folder = st.text_input(
        "Audio Folder Path", 
        placeholder="Enter path to audio files folder",
        help="Full path to the folder containing FLAC audio files"
    )
    
    st.divider()
    
    st.header("🎛️ Spectrogram Settings")
    
    with st.expander("Edit Parameters", expanded=False):
        n_fft = st.slider(
            "FFT Window Size (n_fft)", 
            min_value=256, 
            max_value=4096, 
            value=2048, 
            step=256,
            help="Size of FFT window for spectrogram"
        )
        
        hop_length = st.slider(
            "Hop Length", 
            min_value=64, 
            max_value=1024, 
            value=512, 
            step=64,
            help="Number of samples between successive frames"
        )
        
        z_scale_db = st.slider(
            "Spectrogram dB Range", 
            min_value=30, 
            max_value=120, 
            value=80,
            help="dB range for spectrogram display (dynamic range)"
        )

#state of session (just to save dataframe when app rerun for the new audio)
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "df" not in st.session_state:
    st.session_state.df = None
if "csv_file_name" not in st.session_state:
    st.session_state.csv_file_name = None
if "data_modified" not in st.session_state:
    st.session_state.data_modified = False

def initialize_dataframe(df):
    """Initialize the dataframe with required columns"""
    # initialize 'listened' column (0 = not listened, 1 = listened)
    if 'listened' not in df.columns:
        df['listened'] = 0
    
    # Initialize 'label_listened' column (for false alarm make it -1 or else keep it same as buzzlabel)
    if 'label_listened' not in df.columns:
        df['label_listened'] = df.apply(
            lambda x: -1 if (x['buzzlabel'] == 0 and x['pred_0.95'] == 1) else x['buzzlabel'], 
            axis=1
        )
    
    return df

def save_progress_to_session(df):
    """Save dataframe to session state with backup"""
    st.session_state.df = df.copy()
    st.session_state.data_modified = True

def update_label(df, row_name, column, value):
    """Safely update a label and save to session"""
    df.at[row_name, column] = value
    save_progress_to_session(df)
    return df

def load_audio(audio_path, duration=5):
    """Load audio file and return first 5 seconds"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        # Return first 5 seconds
        return audio[:duration * sr], sr
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None, None

def create_spectrogram(audio, sr, n_fft, hop_length, db_range):
    """Create and return spectrogram plot"""
    # https://librosa.org/doc/0.11.0/generated/librosa.display.specshow.html
    fig, ax = plt.subplots(figsize=(10, 4))
    
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), 
        ref=np.max
    )
    
    # Calculate vmin and vmax based on db_range
    vmax = np.max(D)
    vmin = vmax - db_range
    
    img = librosa.display.specshow(
        D, sr=sr, hop_length=hop_length, 
        x_axis='time', y_axis='hz', 
        ax=ax, vmin=vmin, vmax=vmax,
        cmap='viridis'
    )
    
    ax.set_title("Audio Spectrogram", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    
    cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label("Power (dB)")
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf

if csv_file and audio_folder and os.path.exists(audio_folder):
    current_csv_name = csv_file.name if csv_file else None
    
    if (st.session_state.df is None or 
        st.session_state.csv_file_name != current_csv_name):
        
        st.session_state.df = pd.read_csv(csv_file)
        st.session_state.df = initialize_dataframe(st.session_state.df)
        st.session_state.csv_file_name = current_csv_name
        st.session_state.data_modified = False
        st.session_state.current_index = 0
        st.success("✅ CSV file loaded successfully!")
    
    df = st.session_state.df
    
    # filter for false alarms that haven't been listened to
    false_alarms = df[(df['buzzlabel'] == 0) & (df['pred_0.95'] == 1)]
    unlistened_false_alarms = false_alarms[false_alarms['listened'] == 0]
    
    # stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total False Alarms", len(false_alarms))
    with col2:
        st.metric("Remaining to Review", len(unlistened_false_alarms))
    with col3:
        st.metric("Reviewed", len(false_alarms) - len(unlistened_false_alarms))
    with col4:
        progress = (len(false_alarms) - len(unlistened_false_alarms)) / len(false_alarms) if len(false_alarms) > 0 else 0
        st.metric("Progress", f"{progress:.1%}")
    
    if len(false_alarms) > 0:
        st.progress(progress)
    
    #if all false alarms have been reviewed
    if len(unlistened_false_alarms) == 0:
        st.markdown("""
            <div class="success-message">
                <h3>🎉 Congratulations!</h3>
                <p>All false alarms have been reviewed. You can download the updated CSV file below.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Download button for completed review
        csv_data = df.to_csv(index=False)
        # Create filename with original CSV name
        original_name = os.path.splitext(st.session_state.csv_file_name)[0] if st.session_state.csv_file_name else "data"
        completed_filename = f"completed_audio_review_{original_name}.csv"
        
        st.download_button(
            label="📥 Download Completed Review CSV",
            data=csv_data,
            file_name=completed_filename,
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        if st.session_state.current_index >= len(unlistened_false_alarms):
            st.session_state.current_index = 0
        
        current_row = unlistened_false_alarms.iloc[st.session_state.current_index]
        audio_path = os.path.join(audio_folder, current_row['flacfile'])
        
        st.markdown(f"""
            <div class="audio-info">
                <h3>🎧 Currently Reviewing</h3>
                <p><strong>File:</strong> {current_row['flacfile']}</p>
                <p><strong>Original Buzz Label:</strong> {current_row['buzzlabel']}</p>
                <p><strong>Model Prediction (0.95 threshold):</strong> {current_row['pred_0.95']}</p>
                <p><strong>Review #{len(false_alarms) - len(unlistened_false_alarms) + 1}</strong> of {len(false_alarms)}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists(audio_path):
            audio, sr = load_audio(audio_path)
            
            if audio is not None:
                audio_col, spectro_col = st.columns([1, 1])
                
                with audio_col:
                    st.subheader("🎵 Audio Player")
                    st.audio(audio_path, format='audio/flac')
                    
                    st.subheader("🎯 Your Decision")
                    decision_col1, decision_col2 = st.columns(2)
                    
                    with decision_col1:
                        if st.button("✅ BUZZ DETECTED", type="secondary", use_container_width=True):
                            df = update_label(df, current_row.name, 'label_listened', 1)
                            st.success("✅ Marked as BUZZ")
                            time.sleep(0.5)
                    
                    with decision_col2:
                        if st.button("❌ NO BUZZ", type="secondary", use_container_width=True):
                            df = update_label(df, current_row.name, 'label_listened', 0)
                            st.success("✅ Marked as NO BUZZ")
                            time.sleep(0.5)
                    
                    st.subheader("Change Audio File")
                    nav_col1, nav_col2 = st.columns(2)
                    
                    with nav_col1:
                        if st.button("➡️ NEXT", use_container_width=True):
                            df = update_label(df, current_row.name, 'listened', 1)
                            st.session_state.current_index += 1
                            st.success("➡️ Moving to next audio...")
                            st.rerun()
                    
                    with nav_col2:
                        if st.button("🎲 RANDOM", use_container_width=True):
                            df = update_label(df, current_row.name, 'listened', 1)
                            remaining_indices = list(range(len(unlistened_false_alarms)))
                            if len(remaining_indices) > 1:
                                remaining_indices.remove(st.session_state.current_index)
                                st.session_state.current_index = random.choice(remaining_indices)
                            st.success("🎲 Switching to random audio...")
                            st.rerun()
                
                with spectro_col:
                    st.subheader("📊 Spectrogram")
                    
                    spectro_buf = create_spectrogram(
                        audio, sr, n_fft, hop_length, z_scale_db
                    )
                    st.image(spectro_buf)
                
                st.divider()
                
                if st.session_state.data_modified:
                    st.info("💾 **Data has been modified.** Make sure to download your progress regularly!")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col2:
                    csv_data = df.to_csv(index=False)
                    original_name = os.path.splitext(st.session_state.csv_file_name)[0] if st.session_state.csv_file_name else "data"
                    st.download_button(
                        label="💾 Download Progress",
                        data=csv_data,
                        file_name=f"audio_review_progress_{original_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # with col3:
                #     if st.button("🔄 Force Refresh", use_container_width=True):
                #         st.rerun()
                
                # Display current labels
                with st.expander("🔧 More Information", expanded=False):
                    st.write("Current row data:")
                    current_listened = df.at[current_row.name, 'listened']
                    current_label_listened = df.at[current_row.name, 'label_listened']
                    
                    st.json({
                        'row_index': int(current_row.name),
                        'buzzlabel': int(current_row['buzzlabel']),
                        'pred_0.95': int(current_row['pred_0.95']),
                        'listened': int(current_listened),
                        'label_listened': int(current_label_listened),
                        'data_modified': st.session_state.data_modified,
                        'session_current_index': st.session_state.current_index
                    })
                    
                    # Show summary statistics
                    st.write("Summary Statistics:")
                    stats = {
                        'Total rows': len(df),
                        'False alarms': len(false_alarms),
                        'Listened': len(df[df['listened'] == 1]),
                        'Buzz labels given': len(df[df['label_listened'] == 1]),
                        'No-buzz labels given': len(df[df['label_listened'] == 0])
                    }
                    st.json(stats)
            
            else:
                st.error("Failed to load audio file. Please check the file format and path.")
        
        else:
            st.error(f"Audio file not found: {audio_path}")

elif csv_file and not audio_folder:
    st.warning("Please provide the audio folder path.")

elif not csv_file and audio_folder:
    st.warning("Please upload a CSV file.")

elif csv_file and audio_folder and not os.path.exists(audio_folder):
    st.error("The specified audio folder path does not exist.")

else:
    st.info("""
        👋 **Welcome!**
        
        To get started:
        1. 📁 Upload your CSV file using the sidebar
        2. 📂 Enter the absolute path to your audio files folder in your computer
        3. 🎧 Start reviewing 
        
        This tool will help you manually verify audio predictions where the model detected a buzz 
        but the original label was "no buzz" (false alarms).
    """)
