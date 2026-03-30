"""
knowledge_base.py — Medical knowledge base for NeuroVision RAG pipeline.
Contains hardcoded brain tumor facts. If a PDF/text file is provided,
it is also loaded and chunked automatically.
"""

import os
import re

# -----------------------------
# Hardcoded Medical Knowledge
# -----------------------------
HARDCODED_KNOWLEDGE = [

    # ── Glioma ────────────────────────────────────────────────────────────────
    "Glioma is a type of brain tumor that arises from glial cells, which are the supportive cells of the brain and spinal cord.",
    "Gliomas are classified into low-grade (slow-growing, less aggressive) and high-grade (fast-growing, aggressive) types.",
    "High-grade gliomas such as Glioblastoma Multiforme (GBM) are the most aggressive and have a poor prognosis.",
    "Low-grade gliomas grow slowly and may not cause symptoms for years, but can eventually become high-grade.",
    "Glioma symptoms include persistent headaches, seizures, memory problems, personality changes, and vision or speech difficulties.",
    "Treatment for glioma typically involves surgery to remove as much tumor as possible, followed by radiation therapy and chemotherapy.",
    "Temozolomide is a common chemotherapy drug used to treat glioblastoma after surgery and radiation.",
    "Gliomas account for about 33% of all brain tumors and are more common in adults between 45 and 65 years old.",
    "MRI scans are the primary imaging tool for detecting and monitoring gliomas due to their high soft tissue contrast.",
    "Glioma recurrence is common even after treatment, making regular MRI follow-up essential for monitoring.",

    # ── Meningioma ────────────────────────────────────────────────────────────
    "Meningioma is a tumor that arises from the meninges, the three layers of membranes that surround the brain and spinal cord.",
    "Most meningiomas are benign (non-cancerous) and grow very slowly, but some can be atypical or malignant.",
    "Meningiomas are the most common primary brain tumor, accounting for about 37% of all brain tumors.",
    "Women are more likely to develop meningiomas than men, especially during middle age.",
    "Many meningiomas are discovered incidentally on brain scans done for other reasons and may not need immediate treatment.",
    "Symptoms of meningioma depend on location and include headaches, vision problems, hearing loss, memory loss, and weakness in limbs.",
    "Treatment options for meningioma include observation (watchful waiting), surgery, and radiation therapy.",
    "Radiosurgery such as Gamma Knife is a non-invasive treatment option for small or surgically inaccessible meningiomas.",
    "Meningiomas located near critical brain structures may be difficult to remove completely, increasing recurrence risk.",
    "Exposure to ionizing radiation is a known risk factor for developing meningioma.",

    # ── Pituitary Tumor ───────────────────────────────────────────────────────
    "Pituitary tumors, also called pituitary adenomas, arise from the pituitary gland located at the base of the brain.",
    "The pituitary gland controls many hormones including growth hormone, thyroid-stimulating hormone, and reproductive hormones.",
    "Most pituitary tumors are benign and do not spread to other parts of the body.",
    "Pituitary tumors are classified as functioning (hormone-secreting) or non-functioning (not secreting hormones).",
    "Functioning pituitary tumors can cause hormonal disorders such as Cushing's disease, acromegaly, or hyperprolactinemia.",
    "Cushing's disease is caused by a pituitary tumor that produces excess ACTH, leading to high cortisol levels.",
    "Acromegaly results from a pituitary tumor secreting excess growth hormone, causing abnormal growth of hands, feet, and face.",
    "Symptoms of pituitary tumors include headaches, vision problems (especially loss of peripheral vision), fatigue, and hormonal imbalances.",
    "Treatment for pituitary tumors includes medication, surgery through the nose (transsphenoidal surgery), and radiation therapy.",
    "Dopamine agonists like cabergoline or bromocriptine are effective medications for prolactin-secreting pituitary tumors.",

    # ── No Tumor ──────────────────────────────────────────────────────────────
    "A normal brain MRI shows no evidence of tumor, abnormal masses, or suspicious lesions.",
    "Even with a normal MRI result, patients should follow up with their doctor if symptoms persist.",
    "Normal MRI findings mean the AI model detected no signs of glioma, meningioma, or pituitary tumor in the scan.",
    "A clear MRI scan is reassuring but does not rule out all neurological conditions — clinical evaluation is still important.",

    # ── Grad-CAM Explanations ─────────────────────────────────────────────────
    "Grad-CAM stands for Gradient-weighted Class Activation Mapping, a technique that visualizes which regions of an image influenced the AI model's decision.",
    "In Grad-CAM heatmaps, red and yellow regions indicate areas the model focused on most heavily when making its prediction.",
    "Blue and green regions in a Grad-CAM heatmap indicate areas the model paid little attention to.",
    "Grad-CAM helps doctors and patients understand why the AI made a specific classification, improving transparency and trust.",
    "The Grad-CAM overlay shows the heatmap superimposed on the original MRI scan for easy visual comparison.",
    "Grad-CAM is generated from the last convolutional layer of the neural network, which captures the most high-level visual features.",
    "If Grad-CAM highlights the center or a specific lobe of the brain, it suggests the tumor may be located in that region.",
    "Grad-CAM is an explainability tool — it does not replace a radiologist's interpretation but helps guide attention to suspicious areas.",
    "When no tumor is detected, Grad-CAM is not generated because there is no class activation to visualize.",
    "The NeuroVision system uses VGG16, a deep convolutional neural network, with Grad-CAM to provide visual explanations for its predictions.",

    # ── General Brain MRI Info ────────────────────────────────────────────────
    "MRI (Magnetic Resonance Imaging) uses strong magnetic fields and radio waves to create detailed images of the brain.",
    "Brain MRI is the gold standard imaging technique for detecting brain tumors due to its superior soft tissue contrast.",
    "A contrast-enhanced MRI uses a gadolinium-based dye injected into the bloodstream to highlight abnormal areas more clearly.",
    "MRI scans are non-invasive and do not use ionizing radiation, making them safe for repeated use.",
    "T1-weighted MRI images are useful for seeing brain anatomy, while T2-weighted images highlight fluid and abnormalities.",
    "FLAIR (Fluid-Attenuated Inversion Recovery) MRI sequences are particularly useful for detecting tumors and edema near the brain surface.",
    "Brain tumors can appear as bright or dark spots on MRI depending on their type, water content, and contrast enhancement.",
    "MRI-guided biopsy allows surgeons to precisely sample tumor tissue for laboratory analysis.",
    "Functional MRI (fMRI) can map brain activity and is used before surgery to identify critical brain regions to avoid.",
    "Diffusion Tensor Imaging (DTI) MRI tracks white matter fiber tracts and helps surgeons plan safe routes through the brain.",

    # ── Symptoms & General Info ───────────────────────────────────────────────
    "Common symptoms of brain tumors include persistent headaches (especially in the morning), seizures, nausea, and vomiting.",
    "Cognitive symptoms of brain tumors can include memory loss, difficulty concentrating, and personality or behavior changes.",
    "Motor symptoms such as weakness or numbness in arms or legs may indicate a brain tumor affecting motor pathways.",
    "Vision problems including blurred vision, double vision, or loss of peripheral vision can be caused by brain tumors.",
    "Speech difficulties such as slurred speech or trouble finding words can be signs of a brain tumor in the language areas.",
    "Brain tumor headaches are often worse in the morning, may wake patients from sleep, and worsen with coughing or bending over.",
    "Not all brain tumors cause symptoms, especially slow-growing ones, and may be found incidentally during imaging for other reasons.",
    "Risk factors for brain tumors include family history, exposure to ionizing radiation, and certain genetic syndromes.",
    "Brain tumor diagnosis requires a combination of imaging (MRI/CT), neurological examination, and often a biopsy.",
    "Survival rates for brain tumors vary widely depending on tumor type, grade, location, age, and overall health of the patient.",
    "The NeuroVision AI system is a decision-support tool and should never be used as a substitute for professional medical diagnosis.",
    "Always consult a qualified neurologist or neurosurgeon for proper evaluation and treatment planning after any scan result.",
]


# -----------------------------
# PDF / Text File Loader
# -----------------------------

def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """
    Split a long text into overlapping word-level chunks.
    chunk_size: max words per chunk
    overlap: words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_text_file(filepath: str) -> list[str]:
    """Load a plain .txt file and return chunks."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return _chunk_text(text)


def load_pdf_file(filepath: str) -> list[str]:
    """Load a PDF file and return chunks (requires pypdf)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        full_text = " ".join(
            page.extract_text() or "" for page in reader.pages
        )
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        return _chunk_text(full_text)
    except ImportError:
        print("⚠️  pypdf not installed. Run: pip install pypdf")
        return []
    except Exception as e:
        print(f"⚠️  Could not read PDF {filepath}: {e}")
        return []


def load_external_documents(folder: str = "knowledge_docs") -> list[str]:
    """
    Scan a folder for .txt and .pdf files and return all chunks.
    Create a folder named 'knowledge_docs' in your project root
    and drop any medical PDFs or text files there.
    """
    if not os.path.exists(folder):
        return []

    chunks = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.endswith(".txt"):
            print(f"📄 Loading text file: {filename}")
            chunks.extend(load_text_file(filepath))
        elif filename.endswith(".pdf"):
            print(f"📄 Loading PDF file: {filename}")
            chunks.extend(load_pdf_file(filepath))

    print(f"✅ Loaded {len(chunks)} chunks from external documents")
    return chunks


# -----------------------------
# Combined Knowledge Base
# -----------------------------

def get_all_knowledge() -> list[str]:
    """
    Returns the full knowledge base:
    hardcoded facts + any external documents found in knowledge_docs/
    """
    external = load_external_documents()
    combined = HARDCODED_KNOWLEDGE + external
    print(f"✅ Total knowledge base: {len(combined)} chunks")
    return combined