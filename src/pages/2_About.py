# pages/2_About.py
import streamlit as st

st.set_page_config(page_title="About", layout="wide", initial_sidebar_state="collapsed")


st.title("About This Study")
st.markdown("""
------------------------------------------

AI-generated artworks are becoming extremely sophisticated,  presenting both opportunities and challenges for the creative community.  

#### The Challenge We Face:
With the rapid advancement of AI technologies, distinguishing between AI-generated and human-made artworks has become increasingly difficult. This raises important questions about authenticity, copyrights, and the future of art.
        
#### Why This Matters:
            
The ability to identify AI-generated and human-made artworks is crucial for several reasons:
- **Preserving Artistic Integrity:** Artists deserve recognition and protection for their original work, and consumers have the right to know the true origin of the art they appreciate and purchase.
- **Digital Media and Platforms Regulation:** As AI-generated content becomes more prevalent, maintaining trust in visual media requires reliable detection methods.
- **Informed Decision-Making:** Enabling collectors, galleries, and institutions to make informed decisions about acquisitions and exhibitions.
- **Combating Misinformation:** Preventing the spread of misleading information regarding the origin of artworks.
- **Copyright and Policy:** Understanding image provenance is essential for copyright protection and preventing unauthorised use of creative works.

------------------------------------------
            
#### Explaining Terminology:

- **AI-generated Artwork:** In the context of this study, AI-generated artworks refer to images created by artificial intelligence models (such as Stable Diffusion, Midjourney, or DALL-E) that transform text prompts into images through computational processes.

- **Human-made Artwork:** In the context of this study, human-made artworks refer to images created primarily by a human artist using traditional or digital tools (such as paint, pencil, photography, or digital drawing and editing software), without significant assistance from AI algorithms.

""", unsafe_allow_html=True)
