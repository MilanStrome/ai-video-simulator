import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import easyocr
from sklearn.linear_model import LinearRegression
import pandas as pd
from skimage.metrics import structural_similarity as ssim

st.set_page_config(layout="wide")
st.title("🎯 AI Video Simulation + CTR Predictor + Viewer Personas")

# Step 1: Upload past video data
st.header("Step 1: Add Past Video Data")
custom_keywords = st.text_input("Enter Keywords to Track (comma-separated)", value="baby,toddler,lucas,abc,learn,song,color,number,shape")
keyword_list = [kw.strip().lower() for kw in custom_keywords.split(",") if kw.strip()]

past_titles, past_ctrs, past_features, past_thumbnails = [], [], [], []

for i in range(3):
    st.subheader(f"📼 Past Video {i+1}")
    title = st.text_input(f"Title {i+1}", key=f"ptitle_{i}")
    ctr = st.number_input(f"CTR (%) {i+1}", key=f"pctr_{i}")
    thumbnail = st.file_uploader(f"Thumbnail {i+1}", type=["jpg", "jpeg", "png"], key=f"pthumb_{i}")

    if title and ctr and thumbnail:
        past_titles.append(title)
        past_ctrs.append(ctr)
        image = Image.open(thumbnail)
        resized_image = image.resize((120, 68))  # 20% preview for 600x338 size
        st.image(resized_image, caption=f"Preview of Thumbnail {i+1}", use_container_width=False)
        past_thumbnails.append(image.resize((150,150)))
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = gray.std()
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(np.array(image))
        extracted_text = " ".join([res[1] for res in result])
        text_length = len(extracted_text.strip())

        feature_vector = {f"kw_{kw}": kw in title.lower() for kw in keyword_list}
        feature_vector['brightness'] = brightness
        feature_vector['contrast'] = contrast
        feature_vector['text_length'] = text_length
        past_features.append(feature_vector)

# Train model once all 3 examples are available
model = None
if len(past_features) == 3:
    df_train = pd.DataFrame(past_features)
    X = df_train.values.astype(float)
    y = np.array(past_ctrs).astype(float)
    model = LinearRegression().fit(X, y)
    st.success("✅ Model trained on 3 past videos")

# Step 2: Predict CTR For New Video
div = st.container()
div.header("Step 2: Predict CTR For New Video")
new_title = div.text_input("New Video Title")
new_thumbnail = div.file_uploader("Upload New Thumbnail Image", type=["jpg", "jpeg", "png"], key="newthumb")

if model and new_title and new_thumbnail:
    new_thumbnail_img = Image.open(new_thumbnail)
    gray = cv2.cvtColor(np.array(new_thumbnail_img), cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = gray.std()
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(np.array(new_thumbnail_img))
    extracted_text = " ".join([res[1] for res in result])
    text_length = len(extracted_text.strip())

    feature_vector = pd.DataFrame([{f"kw_{kw}": kw in new_title.lower() for kw in keyword_list} | {
        'brightness': brightness,
        'contrast': contrast,
        'text_length': text_length
    }])

    predicted_ctr = model.predict(feature_vector)[0]
    div.metric("📈 Predicted CTR", f"{predicted_ctr:.2f}%")

    # CTR Benchmark based on kids niche
    if predicted_ctr >= 8:
        div.success("🟢 Excellent CTR! Your video is likely to perform very well.")
    elif predicted_ctr >= 6:
        div.info("🟡 Average CTR. Consider optimizing your title or thumbnail further.")
    else:
        div.warning("🔴 Low CTR. Try improving text clarity, appeal or layout.")

    # --- Compute similarity to past thumbnails ---
    similarity_score = None
    most_similar_index = None

    new_thumb_gray = cv2.cvtColor(np.array(new_thumbnail_img.resize((150,150))), cv2.COLOR_BGR2GRAY)
    similarities = []
    for idx, past_img in enumerate(past_thumbnails):
        past_img_gray = cv2.cvtColor(np.array(past_img), cv2.COLOR_BGR2GRAY)
        score, _ = ssim(new_thumb_gray, past_img_gray, full=True)
        similarities.append(score)

    if similarities:
        similarity_score = max(similarities)
        most_similar_index = similarities.index(similarity_score) + 1  # 1-based

    if similarity_score is not None:
        avg_sim = np.mean(similarities)
        percentage = int(similarity_score * 100)
        most_similar_percentage = int(similarities[most_similar_index - 1] * 100)
        avg_sim_percent = int(avg_sim * 100)

        # st.markdown(f"🖼️ **Thumbnail Visual Similarity to Past Thumbnails:** {percentage}%  \nMost similar to past thumbnail #{most_similar_index} – {most_similar_percentage}%")
        st.markdown(f"📊 **Average Visual Similarity to All Past Thumbnails:** {avg_sim_percent}%")

        max_idx = int(np.argmax(similarities))
        st.markdown(f"**🖼️ Most similar to:** Past Video {max_idx+1} (Similarity: {similarities[max_idx]*100:.1f}%, CTR: {past_ctrs[max_idx]}%)")

                # Visual Side-by-Side
        st.markdown("### 🔍 Side-by-Side Thumbnail Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(new_thumbnail_img.resize((225, 126)), caption="New Thumbnail", use_container_width=False)
        with col2:
            st.image(past_thumbnails[max_idx].resize((225, 126)), caption=f"Most Similar Past Thumbnail #{max_idx+1}", use_container_width=False)


# Step 3: Optional Viewer Simulation
with st.expander("🧠 AI Viewer Personas Simulation"):
    personas = [
        {"name": "Quick Scanner", "age": 3, "engagement_level": "Very Low", "attention_span": 30, "click_keywords": ["fun", "bright", "baby"], "avoid_keywords": ["long", "quiet"]},
        {"name": "Short Peek Viewer", "age": 4, "engagement_level": "Low", "attention_span": 45, "click_keywords": ["colors", "abc", "animals"]},
        {"name": "Curious Toddler", "age": 5, "engagement_level": "Moderate", "attention_span": 90, "click_keywords": ["Lucas", "learn", "song"]},
        {"name": "Steady Watcher", "age": 6, "engagement_level": "Moderate", "attention_span": 120, "click_keywords": ["fun", "count", "interactive"]},
        {"name": "Parent Viewer", "age": 35, "engagement_level": "Low", "attention_span": 60, "click_keywords": ["safe", "toddler", "learning"]},
        {"name": "Teacher Ally", "age": 40, "engagement_level": "High", "attention_span": 180, "click_keywords": ["educational", "structured"]},
        {"name": "Sibling Mia", "age": 8, "engagement_level": "High", "attention_span": 200, "click_keywords": ["challenge", "game"]},
        {"name": "Caregiver Grandma", "age": 65, "engagement_level": "Moderate", "attention_span": 100, "click_keywords": ["classic", "calm"]},
        {"name": "Highly Engaged", "age": 7, "engagement_level": "Very High", "attention_span": 300, "click_keywords": ["Lucas", "story", "music"]},
        {"name": "Quiet Drifter", "age": 3, "engagement_level": "Very Low", "attention_span": 20, "click_keywords": ["soft", "cartoon"]},
    ]

    title = st.text_input("Video Title for Simulation", key="vtitle")
    description = st.text_area("Video Description")
    thumbnail_desc = st.text_area("Thumbnail Description")
    video_script = st.text_area("Video Script Summary")
    uploaded_thumbnail = st.file_uploader("Upload Thumbnail for Persona Check", type=["jpg", "jpeg", "png"], key="simthumb")
    brand_logo = st.file_uploader("Upload Brand Logo (optional)", type=["jpg", "jpeg", "png"], key="brandlogo")

    th_score = 0

    st.sidebar.title("Thumbnail Scoring Thresholds")
    brightness_threshold = st.sidebar.slider("Brightness Threshold", 50, 200, 130)
    contrast_threshold = st.sidebar.slider("Contrast Threshold", 10, 100, 50)
    edge_density_threshold = st.sidebar.slider("Max Edge Density", 0.0, 100.0, 0.05)
    text_length_threshold = st.sidebar.slider("Min Text Characters Detected", 0, 50, 10)
    logo_match_threshold = st.sidebar.slider("Logo Match Score (0-1)", 0.0, 1.0, 0.6)

    if uploaded_thumbnail:
        uploaded_thumbnail.seek(0)
        image_bytes = uploaded_thumbnail.read()
        img_array = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_cv is None or img_cv.size == 0:
            st.error("❌ OpenCV could not read the uploaded image.")
        else:
            image = Image.open(io.BytesIO(image_bytes))
            w, h = image.size
            half_image = image.resize((w // 2, h // 2))
            st.image(half_image, caption="Thumbnail Preview", use_container_width=False)

            try:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                st.error(f"Grayscale error: {e}")
                gray = None

            if gray is not None:
                brightness = np.mean(gray)
                if brightness > brightness_threshold:
                    th_score += 1

                contrast = gray.std()
                if contrast > contrast_threshold:
                    th_score += 1

                th_score += 1  # Simulated face detection pass

                try:
                    reader = easyocr.Reader(['en'], gpu=False)
                    result = reader.readtext(np.array(image))
                    extracted_text = " ".join([res[1] for res in result])
                    st.text_area("🔍 OCR Text from Thumbnail", extracted_text, height=150)
                    if len(extracted_text.strip()) > text_length_threshold:
                        th_score += 1
                except:
                    st.warning("OCR text extract failed")

                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges) / edges.size
                if edge_density < edge_density_threshold:
                    th_score += 1

                if brand_logo:
                    brand_logo.seek(0)
                    logo_bytes = brand_logo.read()
                    logo_array = np.frombuffer(logo_bytes, np.uint8)
                    logo_cv = cv2.imdecode(logo_array, cv2.IMREAD_COLOR)
                    try:
                        result = cv2.matchTemplate(img_cv, logo_cv, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        if max_val > logo_match_threshold:
                            th_score += 1
                            st.info("✅ Brand logo detected.")
                        else:
                            st.warning("⚠️ Brand logo not detected.")
                    except:
                        st.error("Error comparing brand logo.")

                st.markdown(f"**Thumbnail Score: {th_score}/6**")
                if th_score < 3:
                    st.warning("Try improving brightness, contrast, text, face, clarity, or branding.")
                else:
                    st.success("Strong thumbnail!")

    if st.button("Simulate AI Viewers"):
        st.subheader("Simulated Viewer Reactions")
        for p in personas:
            combined_text = f"{title} {description} {thumbnail_desc}".lower()
            keywords_hit = any(kw.lower() in combined_text for kw in p['click_keywords'])
            avoid_hit = any(kw.lower() in combined_text for kw in p.get('avoid_keywords', []))
            boost = th_score >= 3
            clicked = "Yes" if (keywords_hit and not avoid_hit and boost) else "No"
            watch_time = min(p['attention_span'], 180 if clicked == "Yes" else 0)
            reason = f"Clicked due to strong match and thumbnail." if clicked == "Yes" else "Skipped due to mismatch or low appeal."

            st.markdown(f"**{p['name']}** (Age {p['age']})")
            st.markdown(f"- Engagement: **{p['engagement_level']}**")
            st.markdown(f"- Clicked? **{clicked}**")
            st.markdown(f"- Estimated Watch Time: **{watch_time} seconds**")
            st.markdown(f"- Reason: {reason}")
            st.markdown("---")
