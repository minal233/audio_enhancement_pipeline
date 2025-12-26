import streamlit as st
import os
from src.interface.pipeline import enhance_audio_stereo

st.set_page_config(page_title="AI Music Emotion Enhancer", layout="centered")

st.title("🎵 AI-Powered Music Emotion Enhancer")
st.markdown("""
Upload any song → AI predicts its emotional content → Applies intelligent, musical enhancement  
Preserves **full stereo** and **vocal clarity**
""")

uploaded_file = st.file_uploader("Choose an audio file", type=[
                                 'wav', 'mp3', 'ogg', 'flac', 'm4a'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    suffix = uploaded_file.name.split('.')[-1]
    temp_input = f"temp_upload.{suffix}"
    with open(temp_input, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.audio(temp_input)
        st.caption("🎧 Original")

    if st.button("✨ Enhance with AI", type="primary"):
        with st.spinner("Analyzing emotion and enhancing audio..."):
            try:
                output_path, emotions, effect = enhance_audio_stereo(
                    temp_input)

                with col2:
                    st.audio(output_path)
                    st.caption("🔥 AI Enhanced")

                st.success("Enhancement Complete!")

                st.markdown("### Predicted Emotion")
                col_v, col_a = st.columns(2)
                with col_v:
                    st.metric("Valence (Happiness)",
                              f"{emotions['valence']:.3f}")
                with col_a:
                    st.metric("Arousal (Energy)", f"{emotions['arousal']:.3f}")

                st.info(f"**Applied Enhancement:** {effect}")

                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Enhanced Version",
                        data=f,
                        file_name=os.path.basename(output_path),
                        mime="audio/wav"
                    )

            except Exception as e:
                st.error(f"Error during enhancement: {str(e)}")

        # Cleanup temp files
        for file in [temp_input, output_path]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass

st.markdown("---")
st.caption(
    "Built with ❤️ using PyTorch, torchaudio & Streamlit | Model trained on PMEmo dataset")
