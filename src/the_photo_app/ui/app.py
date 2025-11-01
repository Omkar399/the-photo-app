"""Streamlit UI for AFace photo search."""

import streamlit as st
import json
from pathlib import Path
import numpy as np
from PIL import Image
import io

from the_photo_app.config import settings
from the_photo_app.query.search import search_engine

# Set page config
st.set_page_config(
    page_title="🖼️ AFace Photo Search",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main { padding: 2rem; }
    .search-result { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header():
    """Render header."""
    st.title("🖼️ AFace - Local Photo Search")
    st.markdown(
        "Semantic search + face-aware filtering for your local photo library. "
        "**100% private, 100% local.**"
    )
    st.divider()


def render_search_tab():
    """Render text search tab."""
    st.subheader("🔍 Text Search")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g., kids playing soccer at sunset",
            key="text_query",
        )

    with col2:
        topk = st.number_input("Top results", min_value=1, max_value=500, value=3)

    with col3:
        search_button = st.button("🔎 Search", use_container_width=True)

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            alpha = st.slider(
                "Semantic weight (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                help="0 = face-only, 1 = text-only",
            )

        with col2:
            person_filter = st.multiselect(
                "Filter by person",
                ["Alice", "Bob", "Carol"],  # TODO: Load from DB
            )

    # Execute search
    if search_button and query:
        with st.spinner("🔍 Searching..."):
            results = search_engine.search(
                query=query,
                topk=topk,
                alpha=alpha,
                person_labels=person_filter if person_filter else None,
            )

        # Display results
        render_results(results)


def render_face_search_tab():
    """Render face probe search tab."""
    st.subheader("👤 Face Search")

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo containing the face you want to search for",
    )

    if uploaded_file:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(uploaded_file, caption="Probe face", use_column_width=True)

        with col2:
            query = st.text_input(
                "Optional text query",
                placeholder="e.g., beach, park, party...",
                key="face_query",
            )
            topk = st.number_input(
                "Top results",
                min_value=1,
                max_value=500,
                value=3,
                key="face_topk",
            )
            search_button = st.button("🔎 Find Faces", use_container_width=True)

        # Execute search
        if search_button:
            with st.spinner("👤 Searching for similar faces..."):
                try:
                    # Load image
                    probe_image = Image.open(uploaded_file).convert("RGB")
                    probe_array = np.array(probe_image)

                    results = search_engine.probe_face_search(
                        query=query or "",
                        probe_face=probe_array,
                        topk=topk,
                    )

                    render_results(results)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def render_results(results):
    """Render search results."""
    st.divider()

    if not results.get("results"):
        st.info("No results found. Try a different query!")
        return

    count = results.get("count", 0)
    st.success(f"✅ Found {count} results for: **{results.get('query', 'query')}**")

    # Display results in grid
    cols = st.columns(4)
    col_idx = 0

    for result in results.get("results", []):
        with cols[col_idx % 4]:
            with st.container():
                st.divider()

                # Display thumbnail
                image_id = result["image_id"]
                
                # Try different thumbnail paths
                thumb_extensions = ["webp", "jpg", "jpeg", "png"]
                thumb_path = None
                
                for ext in thumb_extensions:
                    candidate = settings.data_dir / "thumbs" / f"{image_id}.{ext}"
                    if candidate.exists():
                        thumb_path = candidate
                        break
                
                if thumb_path:
                    try:
                        # Get appropriate score for caption
                        score_val = result.get('face_score') or result.get('score', 0)
                        st.image(
                            str(thumb_path),
                            use_container_width=True,
                            caption=f"Score: {score_val:.3f}",
                        )
                    except Exception as e:
                        st.warning(f"⚠️ Could not load thumbnail: {e}")
                        st.text(f"ID: {image_id[:16]}...")
                else:
                    st.warning(f"📸 No thumbnail found for {image_id[:16]}...")
                    st.text("(Thumbnail may not have been generated)")

                # Display faces
                faces = result.get("faces", [])
                if faces:
                    st.caption(f"👤 {len(faces)} face(s) detected")
                    for face in faces:
                        person = face.get("person_label") or "Unknown"
                        conf = face.get("confidence", 0)
                        st.text(f"  → {person} ({conf:.2f})")

                # Score/metrics
                if "face_score" in result:
                    st.metric("Face Match", f"{result['face_score']:.3f}")
                else:
                    st.metric("Similarity", f"{result['score']:.3f}")

                col_idx += 1


def render_sidebar():
    """Render sidebar."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Stats
        st.subheader("📊 Index Stats")
        try:
            total_images = search_engine.faiss_mgr.get_index_size()
            st.metric("Images indexed", f"{total_images:,}")
            st.metric("Embedding dim", search_engine.embedding_model.embedding_dim)
            st.metric(
                "Face embedding dim", search_engine.face_detector.embedding_dim
            )
        except Exception as e:
            st.warning(f"Could not load stats: {e}")

        st.divider()

        # Re-indexing
        st.subheader("🔄 Index Management")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Photos in: `{settings.image_dir}`")
        with col2:
            if st.button("🔄", help="Re-index photos", use_container_width=True):
                st.session_state.trigger_reindex = True
        
        if st.session_state.get("trigger_reindex", False):
            with st.spinner("🔄 Re-indexing photos..."):
                try:
                    from the_photo_app.indexing.indexer import PhotoIndexer
                    
                    indexer = PhotoIndexer()
                    
                    # Show progress
                    progress_text = st.empty()
                    progress_text.info("📂 Discovering images...")
                    
                    images = indexer.discover_images(settings.image_dir)
                    progress_text.info(f"📸 Found {len(images)} total images")
                    
                    # Index
                    indexer.index_images(
                        settings.image_dir,
                        batch_size=settings.batch_size,
                        skip_existing=True
                    )
                    
                    progress_text.info("🔄 Reloading search engine...")
                    
                    # Reload the FAISS index in the search engine
                    search_engine.faiss_mgr.load_index()
                    
                    progress_text.empty()
                    st.success("✅ Re-indexing complete!")
                    st.balloons()
                    
                    # Clear trigger
                    st.session_state.trigger_reindex = False
                    
                    # Force refresh to update stats
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Re-indexing failed: {e}")
                    st.session_state.trigger_reindex = False

        st.divider()

        # Information
        st.subheader("ℹ️ Information")
        st.markdown(
            """
        **AFace** is a local-first photo search system featuring:
        
        - 🎯 Semantic search (text → image)
        - 👤 Face detection & recognition
        - 🚀 Sub-second queries with FAISS
        - 🔒 100% private & offline
        
        **Models:**
        - SigLIP for semantic embeddings
        - InsightFace SCRFD + ArcFace for faces
        """
        )

        st.divider()

        # About
        st.markdown("**v0.1.0** | [GitHub](https://github.com/...)")


def main():
    """Main Streamlit app."""
    render_header()
    render_sidebar()

    # Main content
    tab1, tab2 = st.tabs(["🔍 Text Search", "👤 Face Search"])

    with tab1:
        render_search_tab()

    with tab2:
        render_face_search_tab()


if __name__ == "__main__":
    main()
