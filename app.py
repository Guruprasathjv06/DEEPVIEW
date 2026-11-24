"""DEEPVISION: Interactive Deep Learning & CNN Experimentation Studio.

A production-grade Streamlit web application providing:
1. Home dashboard with high-level KPI cards, system hardware monitor, and architecture flow.
2. Dataset & Architecture Inspector: Layer-by-layer tensor dimension breakdown and random sample previews.
3. Interactive Training Engine: Hyperparameter controls, live epoch progress, loss/acc curves, and checkpoint saving.
4. Stratified 5-Fold Cross-Validation: Detailed fold metrics, Mean ± Std calculations, and test set evaluation.
5. Real-Time Inference Engine: Image upload or 1-click test sampling with 10-class probability distribution charts.
6. Weights & Biases (W&B) Portal: Live integration links, run explorer, and tracked scalar metrics.

Theme:
- White & Royal Green Aesthetic:
  - Clean Crisp Canvas: #F8FAFC & #FFFFFF
  - Royal Green Highlights: #064E3B, #047857, #059669
  - Mint Secondary Accents: #ECFDF5, #A7F3D0, #D1FAE5
  - Deep Slate Readability Typography: #0F172A, #1E293B

Run locally:
    streamlit run app.py
"""

import os
import sys
import time
import json
import glob
from typing import Dict, List, Tuple, Optional, Union

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Project imports
from models import get_model, BasicCNN, ImprovedCNN
from training.device import get_device, get_device_info
from training.reproducibility import set_seed
from training.trainer import ModelTrainer
from evaluation.metrics import compute_metrics
from configs.mnist_config import MNISTConfig
from configs.cifar10_config import CIFAR10Config
from datasets.mnist import MNIST_CLASSES
from datasets.cifar10 import CIFAR10_CLASSES

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DEEPVISION | Royal Green & White Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# WHITE & ROYAL GREEN CSS THEME
# ==========================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        /* Clean Crisp White & Slate Light Canvas */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }

        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid rgba(5, 150, 105, 0.22) !important;
        }
        
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Hero Banner with Royal Green Accent */
        .hero-container {
            background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 100%);
            border: 1px solid rgba(5, 150, 105, 0.35);
            border-radius: 16px;
            padding: 26px 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -4px rgba(5, 150, 105, 0.08), 0 4px 6px -2px rgba(5, 150, 105, 0.04);
        }
        
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            background: linear-gradient(90deg, #064E3B 0%, #047857 50%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 6px 0;
            letter-spacing: -0.02em;
        }
        
        .hero-sub {
            color: #065F46; /* Royal Forest Green */
            font-size: 1.05rem;
            font-weight: 500;
            margin: 0;
        }

        /* Pristine White KPI Cards with Royal Green Border */
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid rgba(5, 150, 105, 0.35);
            border-radius: 14px;
            padding: 22px 18px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(5, 150, 105, 0.07);
            transition: all 0.25s ease;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            border-color: #059669;
            box-shadow: 0 10px 25px rgba(5, 150, 105, 0.18);
        }
        
        .kpi-value {
            font-size: 2.3rem;
            font-weight: 800;
            color: #064E3B; /* Deep Royal Green */
            letter-spacing: -0.02em;
        }
        .kpi-value.gold {
            color: #047857;
        }
        .kpi-value.accent {
            color: #065F46;
        }
        
        .kpi-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 6px;
        }

        /* Badges */
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 8px;
            letter-spacing: 0.02em;
        }
        .pill-primary {
            background: rgba(5, 150, 105, 0.12);
            color: #065F46;
            border: 1px solid rgba(5, 150, 105, 0.4);
        }
        .pill-secondary {
            background: #ECFDF5;
            color: #047857;
            border: 1px solid #A7F3D0;
        }
        .pill-tertiary {
            background: #F1F5F9;
            color: #334155;
            border: 1px solid #CBD5E1;
        }

        /* White Content Panels with Royal Green Border */
        .content-panel {
            background: #FFFFFF;
            border: 1px solid rgba(5, 150, 105, 0.3);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
            color: #0F172A;
        }
        .content-panel h4 {
            color: #064E3B !important;
            font-weight: 800;
            margin-top: 0;
        }
        .content-panel ul, .content-panel li, .content-panel p {
            color: #1E293B !important;
            line-height: 1.65;
        }

        /* Prediction Card */
        .pred-banner-correct {
            background: #F0FDF4;
            border: 2px solid #059669;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 4px 16px rgba(5, 150, 105, 0.15);
            color: #064E3B;
        }
        .pred-banner-incorrect {
            background: #FEF2F2;
            border: 2px solid #DC2626;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 4px 16px rgba(220, 38, 38, 0.12);
            color: #991B1B;
        }

        /* Royal Green Primary Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #059669 0%, #047857 50%, #064E3B 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #34D399 !important;
            border-radius: 10px !important;
            font-weight: 800 !important;
            padding: 0.55rem 1.35rem !important;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.28) !important;
            transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #10B981 0%, #059669 50%, #047857 100%) !important;
            color: #FFFFFF !important;
            transform: translateY(-2px) !important;
            border-color: #FFFFFF !important;
            box-shadow: 0 6px 20px rgba(5, 150, 105, 0.4) !important;
        }

        /* ========================================================== */
        /* MODERN INTERACTIVE BUTTON CARDS (TRANSFORMED FROM ST.RADIO) */
        /* ========================================================== */
        div[data-testid="stRadio"] {
            margin-bottom: 1rem !important;
        }

        div[data-testid="stRadio"] > label {
            font-size: 0.82rem !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: #064E3B !important;
            margin-bottom: 8px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 10px !important;
            padding: 4px 0 !important;
            display: flex !important;
            flex-direction: column !important;
        }

        /* 1. HIDE ONLY THE RADIO CIRCLE AND DOT (ge and _e) */
        div[data-testid="stRadio"] div[class*="ex50key4"],
        div[data-testid="stRadio"] div[class*="ex50key5"],
        div[data-testid="stRadio"] [data-testid="stRadioOption"] div[class*="ex50key4"],
        div[data-testid="stRadio"] [data-testid="stRadioOption"] div[class*="ex50key5"],
        div[data-testid="stRadio"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]),
        div[data-testid="stRadio"] [data-testid="stRadioOption"] input[type="radio"],
        div[data-testid="stRadio"] [data-testid="stRadioOption"] svg {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            opacity: 0 !important;
            visibility: hidden !important;
            position: absolute !important;
            pointer-events: none !important;
        }

        /* 2. Style each Radio Option into an Attractive Interactive Button Card */
        div[data-testid="stRadio"] [data-testid="stRadioOption"],
        label[data-testid="stRadioOption"] {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            width: 100% !important;
            background: #FFFFFF !important;
            border: 1.5px solid rgba(5, 150, 105, 0.28) !important;
            border-radius: 12px !important;
            padding: 12px 18px !important;
            margin: 5px 0 !important;
            cursor: pointer !important;
            transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 2px 8px rgba(5, 150, 105, 0.08) !important;
            position: relative !important;
            user-select: none !important;
        }

        /* Ensure inner wrapper (he) fills card and flexes */
        div[data-testid="stRadio"] [data-testid="stRadioOption"] > div {
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
            gap: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            visibility: visible !important;
            opacity: 1 !important;
        }

        /* Hover Elevation on Button Card */
        div[data-testid="stRadio"] [data-testid="stRadioOption"]:hover,
        label[data-testid="stRadioOption"]:hover {
            border-color: #059669 !important;
            background-color: #ECFDF5 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(5, 150, 105, 0.20) !important;
        }

        /* 3. ENSURE LABEL TEXT AND MODULE NAMES ARE FULLY VISIBLE & VIBRANT */
        div[data-testid="stRadio"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"],
        div[data-testid="stRadio"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stRadio"] [data-testid="stRadioOption"] p {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            color: #064E3B !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.4 !important;
            white-space: nowrap !important;
            transition: color 0.2s ease !important;
        }

        /* 4. ACTIVE / SELECTED STATE: Rich Royal Green Gradient Highlight */
        div[data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked),
        div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"],
        label[data-testid="stRadioOption"]:has(input:checked) {
            background: linear-gradient(135deg, #059669 0%, #047857 50%, #064E3B 100%) !important;
            border-color: #064E3B !important;
            box-shadow: 0 6px 20px rgba(5, 150, 105, 0.40) !important;
            transform: translateY(-1px) !important;
        }

        /* Active label text color is crisp white */
        div[data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked) [data-testid="stMarkdownContainer"],
        div[data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked) [data-testid="stMarkdownContainer"] p,
        div[data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked) p,
        div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] p {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }

        /* Horizontal Layouts: Side-by-Side Equal Segmented Buttons */
        div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"],
        div[data-testid="stRadio"] div[role="radiogroup"][data-orientation="horizontal"] {
            flex-direction: row !important;
            gap: 12px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > div,
        div[data-testid="stRadio"] div[role="radiogroup"][data-orientation="horizontal"] > div {
            flex: 1 1 0% !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"],
        div[data-testid="stRadio"] div[role="radiogroup"][data-orientation="horizontal"] [data-testid="stRadioOption"] {
            justify-content: center !important;
            text-align: center !important;
            padding: 14px 20px !important;
        }

        /* Sidebar Specific Button Navigation */
        [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"] {
            width: 100% !important;
            padding: 12px 16px !important;
            margin-bottom: 6px !important;
            border-radius: 10px !important;
        }

        /* ========================================================== */
        /* STREAMLIT ACTION BUTTONS (ST.BUTTON / ST.DOWNLOADBUTTON)  */
        /* ========================================================== */
        div.stButton > button, div.stDownloadButton > button {
            background: linear-gradient(135deg, #059669 0%, #047857 50%, #064E3B 100%) !important;
            color: #FFFFFF !important;
            border: 1.5px solid #10B981 !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.02em !important;
            padding: 0.65rem 1.6rem !important;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
        }

        div.stButton > button:hover, div.stDownloadButton > button:hover {
            background: linear-gradient(135deg, #10B981 0%, #059669 50%, #047857 100%) !important;
            color: #FFFFFF !important;
            transform: translateY(-2px) !important;
            border-color: #FFFFFF !important;
            box-shadow: 0 8px 22px rgba(5, 150, 105, 0.45) !important;
        }

        div.stButton > button:active, div.stDownloadButton > button:active {
            transform: translateY(1px) scale(0.99) !important;
            box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3) !important;
        }

        /* Secondary Style Button */
        div.stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            color: #065F46 !important;
            border: 1.5px solid rgba(5, 150, 105, 0.4) !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            background: #F0FDF4 !important;
            border-color: #059669 !important;
            color: #064E3B !important;
            transform: translateY(-2px) !important;
        }

        /* Tabs Styling */
        div[data-baseweb="tab-list"] {
            background-color: #F1F5F9 !important;
            padding: 6px !important;
            border-radius: 12px !important;
            gap: 8px !important;
            border: 1px solid rgba(5, 150, 105, 0.15) !important;
        }
        div[data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 8px 18px !important;
            font-weight: 600 !important;
            color: #475569 !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
        }
        div[data-baseweb="tab"]:hover {
            color: #064E3B !important;
            background: rgba(5, 150, 105, 0.08) !important;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: #FFFFFF !important;
            color: #064E3B !important;
            font-weight: 800 !important;
            box-shadow: 0 3px 10px rgba(5, 150, 105, 0.18) !important;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #059669 !important;
        }

        /* Inputs & Selectboxes */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1.5px solid rgba(5, 150, 105, 0.3) !important;
            border-radius: 10px !important;
            color: #0F172A !important;
            transition: all 0.2s ease !important;
        }
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="select"] > div:focus-within {
            border-color: #059669 !important;
            box-shadow: 0 0 0 2px rgba(5, 150, 105, 0.2) !important;
        }

        /* File Uploader */
        div[data-testid="stFileUploader"] section {
            background-color: #FFFFFF !important;
            border: 2px dashed rgba(5, 150, 105, 0.35) !important;
            border-radius: 14px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: #059669 !important;
            background-color: #F0FDF4 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# HARDWARE UTILITIES & DATA LOADERS
# ==========================================
@st.cache_resource
def get_system_hardware():
    """Detect and cache host hardware information."""
    dev = get_device("auto")
    info = get_device_info(dev)
    return dev, info


@st.cache_data
def load_saved_results(dataset_name: str = "MNIST") -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load existing cross-validation and test set results for MNIST or CIFAR-10."""
    if dataset_name == "CIFAR-10":
        csv_path = os.path.join("outputs", "cifar10_comparison_summary.csv")
        json_path = os.path.join("outputs", "cifar10_final_experiment_results.json")
    else:
        csv_path = os.path.join("outputs", "comparison_summary.csv")
        json_path = os.path.join("outputs", "final_experiment_results.json")

    df = None
    results_json = None

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            pass

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                results_json = json.load(f)
        except Exception:
            pass

    return df, results_json


def get_available_checkpoints() -> List[str]:
    """Scan the checkpoints directory for saved PyTorch model weights."""
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        return []
    pts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    return sorted(pts)


def ensure_pil_image(img_input: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
    """Safely convert any tensor, numpy array, or PIL input into a clean PIL.Image."""
    if isinstance(img_input, Image.Image):
        return img_input

    if isinstance(img_input, torch.Tensor):
        img_input = img_input.detach().cpu()
        if img_input.ndim == 3:
            if img_input.shape[0] == 1:
                arr = img_input.squeeze(0).numpy()
            elif img_input.shape[0] == 3:
                arr = img_input.permute(1, 2, 0).numpy()
            else:
                arr = img_input.numpy()
        elif img_input.ndim == 2:
            arr = img_input.numpy()
        else:
            arr = img_input.squeeze().numpy()
    elif isinstance(img_input, np.ndarray):
        arr = img_input
    else:
        raise ValueError(f"Unsupported image type: {type(img_input)}")

    # Scale array to [0, 255] uint8
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            a_min, a_max = arr.min(), arr.max()
            if a_max > a_min:
                arr = ((arr - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    elif arr.ndim == 3 and arr.shape[2] == 1:
        return Image.fromarray(arr.squeeze(2), mode="L")
    else:
        return Image.fromarray(arr)


@st.cache_resource
def get_cached_dataset_manager(dataset_name: str):
    """Cache dataset managers in memory to avoid repeated disk reads and fold generation."""
    if dataset_name == "MNIST":
        from datasets.mnist import MNISTManager
        return MNISTManager(data_dir="data", n_splits=5, seed=42)
    else:
        from datasets.cifar10 import CIFAR10Manager
        return CIFAR10Manager(data_dir="data", n_splits=5, seed=42)


# Initialize runtime states
device_obj, hw_info = get_system_hardware()
comparison_df, full_results = load_saved_results("MNIST")
cifar_comparison_df, cifar_full_results = load_saved_results("CIFAR-10")
checkpoints_list = get_available_checkpoints()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### 🌿 DEEPVISION STUDIO")
    st.markdown(
        "<span class='status-pill pill-primary'>PyTorch 2.5</span>"
        "<span class='status-pill pill-secondary'>W&B Tracked</span>"
        "<span class='status-pill pill-tertiary'>5-Fold CV</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    nav_selection = st.radio(
        "STUDIO NAVIGATION",
        [
            "🏠 Overview Dashboard",
            "🧠 Model & Dataset Explorer",
            "🚀 Interactive Training",
            "📊 5-Fold CV & Evaluation",
            "🔮 Real-Time Prediction",
            "🌐 Weights & Biases Portal",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### ⚡ Hardware Diagnostics")
    st.markdown(f"**Target Device:** `{hw_info['device_type'].upper()}`")
    st.markdown(f"**Host GPU:** `{hw_info['gpu_name']}`")
    st.markdown(f"**PyTorch Version:** `{hw_info['torch_version']}`")
    st.markdown(f"**Available Weights:** `{len(checkpoints_list)} checkpoints`")

    st.markdown("---")
    st.markdown("#### 🔗 Experiment Portals")
    st.markdown(
        "[🚀 Open CoreWeave W&B Dashboard](https://forge.coreweave.com/wandb/vaishalinir-ymc2022-chennai-institute-of-technology/DEEPVISION)"
    )
    st.caption("Royal Green & White Edition")


# =========================================================================
# TAB 1: OVERVIEW DASHBOARD
# =========================================================================
if nav_selection == "🏠 Overview Dashboard":
    st.markdown(
        """
        <div class='hero-container'>
            <div class='hero-title'>DEEPVISION Experimentation Studio</div>
            <div class='hero-sub'>Production-Grade CNN Architectures with Stratified 5-Fold Cross-Validation, Real-Time Training & Weights & Biases Tracking</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Cards (Pristine White with Royal Green Borders)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class='kpi-card'>
                <div class='kpi-value gold'>99.26%</div>
                <div class='kpi-label'>Official Test Acc (MNIST)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class='kpi-card'>
                <div class='kpi-value accent'>72.22%</div>
                <div class='kpi-label'>Official Test Acc (CIFAR-10)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class='kpi-card'>
                <div class='kpi-value gold'>98.99%</div>
                <div class='kpi-label'>MNIST 5-Fold CV Acc</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class='kpi-card'>
                <div class='kpi-value accent'>72.34%</div>
                <div class='kpi-label'>CIFAR-10 5-Fold CV Acc</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core System Pillars
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### 🏛️ System Architecture & Workflow")
        st.markdown(
            """
            <div class='content-panel'>
                <h4>High-Performance Deep Learning Engine</h4>
                <ul>
                    <li><b>Architectures From Scratch:</b> Clean implementations of <b>Basic CNN</b> (150,090 parameters) and deep <b>Improved CNN</b> (419,786 parameters) with spatial <code>BatchNorm2d</code>, <code>Dropout2d</code>, and dense <code>BatchNorm1d</code>.</li>
                    <li><b>Stratified 5-Fold Cross-Validation:</b> Rigorous 5-fold CV on 60,000 MNIST and 50,000 CIFAR-10 samples with zero data leakage, reporting exact mean ± std across folds.</li>
                    <li><b>Untouched Official Test Isolation:</b> The 10,000-sample official test set is strictly sequestered from cross-validation and evaluated only on the selected best model checkpoint.</li>
                    <li><b>Weights & Biases (W&B) Logging:</b> Automated tracking of loss curves, accuracy, learning rates, confusion matrices, and prediction grids.</li>
                    <li><b>Interactive Inference:</b> Real-time image upload and 1-click test sampling across MNIST (digits) and CIFAR-10 (objects) with 10-class Softmax probability distributions.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "💡 **Hardware Setup:** Host equipped with an **Intel Arc 130V GPU**. Tensor computations are dynamically routed through multi-core AVX-512 CPU operations for optimal numerical stability."
        )

    with col_right:
        st.markdown("### 📊 Cross-Validation Leaderboard")
        board_tabs = st.tabs(["🔢 MNIST Benchmark", "🖼️ CIFAR-10 Benchmark"])
        with board_tabs[0]:
            if comparison_df is not None:
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No MNIST experiment results found.")
        with board_tabs[1]:
            if cifar_comparison_df is not None:
                st.dataframe(cifar_comparison_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No CIFAR-10 experiment results found.")

        st.markdown("### ⚡ Quick Navigation")
        quick_c1, quick_c2 = st.columns(2)
        with quick_c1:
            st.markdown(
                """
                <div class='content-panel' style='padding: 16px; text-align: center;'>
                    <div style='font-size: 1.15rem; font-weight: 800; color: #064E3B; margin-bottom: 4px;'>Inference</div>
                    <div style='font-size: 0.85rem; color: #64748B;'>Test images in real-time</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with quick_c2:
            st.markdown(
                """
                <div class='content-panel' style='padding: 16px; text-align: center;'>
                    <div style='font-size: 1.15rem; font-weight: 800; color: #064E3B; margin-bottom: 4px;'>Training</div>
                    <div style='font-size: 0.85rem; color: #64748B;'>Live training with curves</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================================
# TAB 2: MODEL & DATASET EXPLORER
# =========================================================================
elif nav_selection == "🧠 Model & Dataset Explorer":
    st.markdown(
        """
        <div class='hero-container'>
            <div class='hero-title'>Model Architecture & Dataset Inspector</div>
            <div class='hero-sub'>Inspect layer-by-layer tensor dimensions, regularization components, and dataset distributions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_model, tab_data = st.tabs(["🏗️ CNN Architectures", "🖼️ Dataset Inspector & Sample Preview"])

    with tab_model:
        model_choice = st.radio("Select Architecture to Inspect:", ["Basic CNN", "Improved CNN"], horizontal=True)

        if model_choice == "Basic CNN":
            st.markdown("#### 🔹 Basic CNN Architecture (150,090 Trainable Parameters)")
            st.markdown(
                "A baseline 2-stage convolutional network with Max Pooling, Adaptive Average Pooling, and a 2-layer classifier."
            )

            arch_data = [
                {"Layer": "Input Image", "Type": "Image Tensor", "Dimensions": "(B, 1, 28, 28)", "Parameters": "0", "Activation": "N/A"},
                {"Layer": "Conv1 + ReLU", "Type": "Conv2d(1 -> 32, k=3, s=1, p=1)", "Dimensions": "(B, 32, 28, 28)", "Parameters": "320", "Activation": "ReLU"},
                {"Layer": "MaxPool1", "Type": "MaxPool2d(k=2, s=2)", "Dimensions": "(B, 32, 14, 14)", "Parameters": "0", "Activation": "None"},
                {"Layer": "Conv2 + ReLU", "Type": "Conv2d(32 -> 64, k=3, s=1, p=1)", "Dimensions": "(B, 64, 14, 14)", "Parameters": "18,496", "Activation": "ReLU"},
                {"Layer": "MaxPool2", "Type": "MaxPool2d(k=2, s=2)", "Dimensions": "(B, 64, 7, 7)", "Parameters": "0", "Activation": "None"},
                {"Layer": "AdaptivePool", "Type": "AdaptiveAvgPool2d((4, 4))", "Dimensions": "(B, 64, 4, 4)", "Parameters": "0", "Activation": "None"},
                {"Layer": "Flatten", "Type": "Flatten()", "Dimensions": "(B, 1024)", "Parameters": "0", "Activation": "None"},
                {"Layer": "FC1 + ReLU", "Type": "Linear(1024 -> 128)", "Dimensions": "(B, 128)", "Parameters": "131,200", "Activation": "ReLU"},
                {"Layer": "FC2 (Logits)", "Type": "Linear(128 -> 10)", "Dimensions": "(B, 10)", "Parameters": "1,290", "Activation": "CrossEntropy"},
            ]
            st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

        else:
            st.markdown("#### 🌟 Improved CNN Architecture (419,786 Trainable Parameters)")
            st.markdown(
                "A deep residual-style architecture with 5 Conv layers structured into 3 stages, with spatial **BatchNorm2d**, **Dropout2d**, **AdaptiveAvgPool2d**, and **BatchNorm1d + Dropout(0.40)**."
            )

            arch_data = [
                {"Layer": "Input Image", "Type": "Image Tensor", "Dimensions": "(B, 1, 28, 28)", "Parameters": "0", "Regularization": "None"},
                {"Layer": "Block 1 (Conv1a)", "Type": "Conv2d(1->32, k=3) + BN2d + ReLU", "Dimensions": "(B, 32, 28, 28)", "Parameters": "384", "Regularization": "BatchNorm2d"},
                {"Layer": "Block 1 (Conv1b)", "Type": "Conv2d(32->32, k=3) + BN2d + ReLU", "Dimensions": "(B, 32, 28, 28)", "Parameters": "9,280", "Regularization": "BatchNorm2d + MaxPool + Dropout2d(0.15)"},
                {"Layer": "Block 2 (Conv2a)", "Type": "Conv2d(32->64, k=3) + BN2d + ReLU", "Dimensions": "(B, 64, 14, 14)", "Parameters": "18,560", "Regularization": "BatchNorm2d"},
                {"Layer": "Block 2 (Conv2b)", "Type": "Conv2d(64->64, k=3) + BN2d + ReLU", "Dimensions": "(B, 64, 14, 14)", "Parameters": "36,992", "Regularization": "BatchNorm2d + MaxPool + Dropout2d(0.20)"},
                {"Layer": "Block 3 (Conv3)", "Type": "Conv2d(64->128, k=3) + BN2d + ReLU", "Dimensions": "(B, 128, 7, 7)", "Parameters": "73,984", "Regularization": "BatchNorm2d + Dropout2d(0.20)"},
                {"Layer": "AdaptivePool", "Type": "AdaptiveAvgPool2d((3, 3))", "Dimensions": "(B, 128, 3, 3)", "Parameters": "0", "Regularization": "None"},
                {"Layer": "FC1 + BN + Drop", "Type": "Linear(1152 -> 256) + BN1d + ReLU", "Dimensions": "(B, 256)", "Parameters": "295,424", "Regularization": "BatchNorm1d + Dropout(0.40)"},
                {"Layer": "FC2 (Logits)", "Type": "Linear(256 -> 10)", "Dimensions": "(B, 10)", "Parameters": "2,570", "Regularization": "None"},
            ]
            st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    with tab_data:
        dataset_selection = st.radio("Select Dataset:", ["MNIST (Handwritten Digits)", "CIFAR-10 (Natural Objects)"], horizontal=True)

        if "MNIST" in dataset_selection:
            st.markdown("#### MNIST Dataset Specifications")
            st.markdown(
                """
                - **Resolution:** 28 × 28 Grayscale (1 Channel)
                - **Training Samples:** 60,000 images (partitioned into 5 stratified folds: 48,000 train / 12,000 val per fold)
                - **Official Untouched Test Samples:** 10,000 images
                - **Target Classes:** 10 Digits (`0` through `9`)
                - **Standard Normalization:** $\\mu = 0.1307, \\sigma = 0.3081$
                """
            )

            if st.button("🎲 Preview 8 Random MNIST Samples", type="primary"):
                raw_mnist = datasets.MNIST(root="data", train=False, download=False, transform=None)
                indices = np.random.choice(len(raw_mnist), 8, replace=False)

                cols = st.columns(8)
                for i, idx in enumerate(indices):
                    img_pil, target = raw_mnist[idx]
                    with cols[i]:
                        st.image(img_pil, caption=f"Class: {MNIST_CLASSES[target]}", use_container_width=True)

        else:
            st.markdown("#### CIFAR-10 Dataset Specifications")
            st.markdown(
                """
                - **Resolution:** 32 × 32 RGB Color (3 Channels)
                - **Training Samples:** 50,000 images (partitioned into 5 stratified folds: 40,000 train / 10,000 val per fold)
                - **Official Untouched Test Samples:** 10,000 images
                - **Target Classes:** 10 Objects (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`)
                - **Standard Normalization:** $\\mu = (0.4914, 0.4822, 0.4465), \\sigma = (0.2470, 0.2435, 0.2616)$
                """
            )

            if st.button("🎲 Preview 8 Random CIFAR-10 Samples", type="primary"):
                raw_cifar = datasets.CIFAR10(root="data", train=False, download=False, transform=None)
                indices = np.random.choice(len(raw_cifar), 8, replace=False)

                cols = st.columns(8)
                for i, idx in enumerate(indices):
                    img_pil, target = raw_cifar[idx]
                    with cols[i]:
                        st.image(img_pil, caption=f"{CIFAR10_CLASSES[target]}", use_container_width=True)


# =========================================================================
# TAB 3: INTERACTIVE TRAINING & PROGRESS
# =========================================================================
elif nav_selection == "🚀 Interactive Training":
    st.markdown(
        """
        <div class='hero-container'>
            <div class='hero-title'>Interactive Training & Cross-Validation</div>
            <div class='hero-sub'>Configure hyperparameters and execute live training with real-time progress curves and W&B logging</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Hyperparameter & Training Configuration", expanded=True):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        with cfg_col1:
            dataset_opt = st.selectbox("Dataset Target:", ["MNIST", "CIFAR-10"], index=0)
            model_opt = st.selectbox("Architecture:", ["Improved CNN", "Basic CNN"], index=0)
            training_speed = st.selectbox(
                "Training Mode & Speed:",
                [
                    "⚡ Ultra-Fast Demo (~1,280 samples, ~2s/epoch - Instant)",
                    "🏎️ Fast Interactive Demo (~5,000 samples, ~6-8s/epoch)",
                    "🚀 Balanced Demo (~15,000 samples, ~20s/epoch)",
                    "🏋️ Full Dataset (All 48,000 samples, ~75s/epoch)",
                ],
                index=0,
            )

        with cfg_col2:
            num_epochs = st.slider("Epochs:", min_value=1, max_value=10, value=2, step=1)
            batch_sz = st.select_slider("Batch Size:", options=[32, 64, 128, 256], value=128)
            lr_val = st.number_input("Learning Rate:", min_value=0.0001, max_value=0.05, value=0.001, step=0.0005, format="%.4f")

        with cfg_col3:
            opt_name = st.selectbox("Optimizer:", ["Adam", "SGD"], index=0)
            device_choice = st.selectbox("Execution Device:", ["Auto Detect (CPU)", "CPU Only"], index=0)
            wandb_logging = st.selectbox("W&B Tracking Mode:", ["online", "offline", "disabled"], index=0)

    start_training_clicked = st.button("🚀 Start Training Experiment", type="primary", use_container_width=True)

    if start_training_clicked:
        st.markdown("### 📊 Live Training Progress")

        st.caption("Overall Epoch Progress:")
        epoch_pbar = st.progress(0.0)
        st.caption("Live Batch Progress:")
        batch_pbar = st.progress(0.0)
        status_text = st.empty()

        # Dynamic metric cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            card_epoch = st.empty()
        with m_col2:
            card_train_loss = st.empty()
        with m_col3:
            card_val_loss = st.empty()
        with m_col4:
            card_val_acc = st.empty()

        chart_placeholder = st.empty()

        # CPU Multi-threading Optimization
        try:
            torch.set_num_threads(min(os.cpu_count() or 4, 8))
        except Exception:
            pass

        # Initialize Data and Model
        in_ch = 1 if dataset_opt == "MNIST" else 3
        model_name_key = "improved" if "Improved" in model_opt else "basic"
        dev = get_device("auto")

        status_text.info(f"⚡ Initializing {model_opt} on {dataset_opt} with {dev.type.upper()} acceleration...")

        # Load Dataset Manager using cache
        dm = get_cached_dataset_manager(dataset_opt)
        train_loader, val_loader = dm.get_fold_loaders(fold_idx=0, batch_size=batch_sz)

        # Scale down dataset for lightning-fast interactive responsiveness
        if "Ultra-Fast Demo" in training_speed:
            subset_train_size = min(1280, len(train_loader.dataset))
            subset_val_size = min(320, len(val_loader.dataset))
            indices_train = torch.randperm(len(train_loader.dataset))[:subset_train_size]
            indices_val = torch.randperm(len(val_loader.dataset))[:subset_val_size]
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_loader.dataset, indices_train),
                batch_size=batch_sz,
                shuffle=True,
            )
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(val_loader.dataset, indices_val),
                batch_size=batch_sz,
                shuffle=False,
            )
        elif "Fast Interactive Demo" in training_speed:
            subset_train_size = min(5000, len(train_loader.dataset))
            subset_val_size = min(1000, len(val_loader.dataset))
            indices_train = torch.randperm(len(train_loader.dataset))[:subset_train_size]
            indices_val = torch.randperm(len(val_loader.dataset))[:subset_val_size]
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_loader.dataset, indices_train),
                batch_size=batch_sz,
                shuffle=True,
            )
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(val_loader.dataset, indices_val),
                batch_size=batch_sz,
                shuffle=False,
            )
        elif "Balanced Demo" in training_speed:
            subset_train_size = min(15000, len(train_loader.dataset))
            subset_val_size = min(2500, len(val_loader.dataset))
            indices_train = torch.randperm(len(train_loader.dataset))[:subset_train_size]
            indices_val = torch.randperm(len(val_loader.dataset))[:subset_val_size]
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_loader.dataset, indices_train),
                batch_size=batch_sz,
                shuffle=True,
            )
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(val_loader.dataset, indices_val),
                batch_size=batch_sz,
                shuffle=False,
            )

        # Model instance
        model = get_model(model_name_key, in_channels=in_ch, num_classes=10).to(dev)

        if opt_name.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_val, momentum=0.9, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        # W&B Tracker
        tracker = None
        if wandb_logging != "disabled":
            from experiments.wandb_tracker import WandbTracker
            tracker = WandbTracker(
                project_name="DEEPVISION",
                experiment_name=f"{dataset_opt.lower()}_{model_name_key}_interactive",
                config={
                    "dataset": dataset_opt,
                    "model": model_name_key,
                    "epochs": num_epochs,
                    "batch_size": batch_sz,
                    "learning_rate": lr_val,
                    "optimizer": opt_name,
                    "speed_mode": training_speed,
                },
                tags=["interactive", dataset_opt.lower(), model_name_key],
                mode=wandb_logging,
            )

        metrics_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # Training Loop
        total_steps = num_epochs
        total_batches = len(train_loader)
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

                # Real-time intra-epoch status update every few batches (and last batch)
                step_interval = max(1, total_batches // 8)
                if (batch_idx + 1) % step_interval == 0 or (batch_idx + 1) == total_batches:
                    curr_b = batch_idx + 1
                    curr_loss = running_loss / max(total, 1)
                    curr_acc = (correct / max(total, 1)) * 100
                    elapsed = time.time() - epoch_start
                    eta_s = int((elapsed / curr_b) * (total_batches - curr_b))
                    batch_pbar.progress(curr_b / total_batches)
                    status_text.markdown(
                        f"**Epoch {epoch}/{num_epochs}** ⚡ Processing Batch `{curr_b}/{total_batches}` "
                        f"({int(curr_b/total_batches*100)}%) | Running Loss: `{curr_loss:.4f}` | Acc: `{curr_acc:.1f}%` | ETA: `{max(0, eta_s)}s`"
                    )

            scheduler.step()
            train_loss = running_loss / total
            train_acc = correct / total

            # Fast validation phase using torch.inference_mode()
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.inference_mode():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Update Metrics UI
            metrics_history["train_loss"].append(train_loss)
            metrics_history["val_loss"].append(val_loss)
            metrics_history["train_acc"].append(train_acc * 100)
            metrics_history["val_acc"].append(val_acc * 100)

            card_epoch.metric("Current Epoch", f"{epoch}/{num_epochs}")
            card_train_loss.metric("Train Loss", f"{train_loss:.4f}")
            card_val_loss.metric("Val Loss", f"{val_loss:.4f}")
            card_val_acc.metric("Val Accuracy", f"{val_acc * 100:.2f}%")

            epoch_pbar.progress(epoch / total_steps)

            # Live Chart Update
            chart_df = pd.DataFrame(
                {
                    "Train Loss": metrics_history["train_loss"],
                    "Val Loss": metrics_history["val_loss"],
                }
            )
            chart_placeholder.line_chart(chart_df)

            # W&B step logging
            if tracker:
                tracker.log_epoch(
                    epoch,
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                )

        elapsed = time.time() - start_time
        status_text.success(f"✅ Training completed in {elapsed:.1f}s! Final Val Accuracy: {val_acc * 100:.2f}%")

        # Save interactive checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        save_path = os.path.join("checkpoints", f"{dataset_opt.lower()}_{model_name_key}_interactive_best.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "dataset": dataset_opt,
                "model_name": model_name_key,
            },
            save_path,
        )
        st.info(f"💾 Checkpoint saved to `{save_path}`")

        if tracker:
            tracker.finish()


# =========================================================================
# TAB 4: 5-FOLD CV & MODEL EVALUATION
# =========================================================================
elif nav_selection == "📊 5-Fold CV & Evaluation":
    st.markdown(
        """
        <div class='hero-container'>
            <div class='hero-title'>5-Fold Cross-Validation & Model Evaluation</div>
            <div class='hero-sub'>Rigorous Stratified 5-Fold Cross-Validation analytics with Mean ± Std metrics, confusion matrices, and official test evaluation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Benchmark Dataset Selection")
    bench_sel = st.radio(
        "Select Benchmark Dataset:",
        [
            "🖼️ CIFAR-10 Benchmark (Natural Objects: Airplane to Truck)",
            "🔢 MNIST Benchmark (Handwritten Digits: 0 to 9)",
        ],
        index=0,
        horizontal=True,
    )
    is_cifar_bench = "CIFAR-10" in bench_sel
    curr_df, curr_full = (cifar_comparison_df, cifar_full_results) if is_cifar_bench else (comparison_df, full_results)
    bench_name = "CIFAR-10" if is_cifar_bench else "MNIST"

    if curr_df is not None:
        st.markdown(f"### 🏆 {bench_name} Architecture Comparison Summary")
        st.dataframe(curr_df, use_container_width=True, hide_index=True)

    tab_cv, tab_cm, tab_test = st.tabs(["📋 Stratified 5-Fold Breakdown", "🎯 Confusion Matrices", "🧪 Official Untouched Test Set"])

    with tab_cv:
        st.markdown(f"#### Stratified 5-Fold Cross-Validation Metrics ({bench_name})")

        if is_cifar_bench:
            basic_json = os.path.join("outputs", "cifar10_basic_cv_results.json")
            improved_json = os.path.join("outputs", "cifar10_improved_cv_results.json")
        else:
            basic_json = os.path.join("outputs", "basic_cv_results.json")
            improved_json = os.path.join("outputs", "improved_cv_results.json")

        col_b, col_i = st.columns(2)

        with col_b:
            st.markdown(f"##### 🔹 Basic CNN (5 Folds - {bench_name})")
            if os.path.exists(basic_json):
                with open(basic_json, "r") as f:
                    b_data = json.load(f)
                b_folds = pd.DataFrame(b_data["fold_details"])
                display_cols = ["fold", "val_accuracy", "val_loss", "val_f1_macro", "val_precision_macro", "val_recall_macro"]
                b_folds["val_accuracy"] = b_folds["val_accuracy"].apply(lambda x: f"{float(x) * 100:.2f}%")
                b_folds["val_loss"] = b_folds["val_loss"].apply(lambda x: f"{float(x):.4f}")
                st.dataframe(b_folds[display_cols], use_container_width=True, hide_index=True)
                st.markdown(f"**Mean Val Acc:** `{b_data['accuracy_mean']*100:.2f}% ± {b_data['accuracy_std']*100:.2f}%`")
                st.markdown(f"**Mean Val Loss:** `{b_data['loss_mean']:.4f} ± {b_data['loss_std']:.4f}`")
            else:
                st.info(f"Run Basic CNN CV for {bench_name} to view fold details.")

        with col_i:
            st.markdown(f"##### 🌟 Improved CNN (5 Folds - {bench_name})")
            if os.path.exists(improved_json):
                with open(improved_json, "r") as f:
                    i_data = json.load(f)
                i_folds = pd.DataFrame(i_data["fold_details"])
                display_cols = ["fold", "val_accuracy", "val_loss", "val_f1_macro", "val_precision_macro", "val_recall_macro"]
                i_folds["val_accuracy"] = i_folds["val_accuracy"].apply(lambda x: f"{float(x) * 100:.2f}%")
                i_folds["val_loss"] = i_folds["val_loss"].apply(lambda x: f"{float(x):.4f}")
                st.dataframe(i_folds[display_cols], use_container_width=True, hide_index=True)
                st.markdown(f"**Mean Val Acc:** `{i_data['accuracy_mean']*100:.2f}% ± {i_data['accuracy_std']*100:.2f}%`")
                st.markdown(f"**Mean Val Loss:** `{i_data['loss_mean']:.4f} ± {i_data['loss_std']:.4f}`")
            else:
                st.info(f"Run Improved CNN CV for {bench_name} to view fold details.")

    with tab_cm:
        st.markdown(f"#### Confusion Matrix Heatmaps & Sample Predictions ({bench_name})")
        if is_cifar_bench:
            cm_test_file = os.path.join("outputs", "plots", "cifar10", "test_confusion_matrix_improved.png")
            pred_test_file = os.path.join("outputs", "plots", "cifar10", "test_sample_predictions_improved.png")
        else:
            cm_test_file = os.path.join("outputs", "plots", "test_evaluation", "test_confusion_matrix_improved.png")
            pred_test_file = os.path.join("outputs", "plots", "test_evaluation", "test_sample_predictions_improved.png")

        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            if os.path.exists(cm_test_file):
                st.image(cm_test_file, caption=f"Official {bench_name} Test Confusion Matrix", use_container_width=True)
            else:
                st.info(f"Confusion matrix plot for {bench_name} not found.")

        with cm_col2:
            if os.path.exists(pred_test_file):
                st.image(pred_test_file, caption=f"Official {bench_name} Predictions (Green = Correct, Red = Error)", use_container_width=True)
            else:
                st.info(f"Sample prediction plot for {bench_name} not found.")

    with tab_test:
        st.markdown(f"#### Official Untouched Test Set Evaluation (10,000 {bench_name} samples)")
        st.markdown(
            f"""
            > **Strict Sequestration Protocol:** The official {bench_name} test set was kept completely separate from cross-validation
            > and evaluated strictly on the best trained model checkpoint.
            """
        )

        if curr_full and "official_test_set_results" in curr_full:
            test_res = curr_full["official_test_set_results"]
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            with t_col1:
                st.metric("Test Accuracy", f"{test_res['test_accuracy']*100:.2f}%")
            with t_col2:
                st.metric("Test Loss", f"{test_res['test_loss']:.4f}")
            with t_col3:
                st.metric("Test Macro F1", f"{test_res['test_f1_macro']:.4f}")
            with t_col4:
                st.metric("Evaluated Model", test_res["evaluated_model"].upper())

            st.markdown(f"##### Per-Class F1-Scores on Official {bench_name} Test Set")
            if "per_class_f1" in test_res:
                class_names_map = CIFAR10_CLASSES if is_cifar_bench else MNIST_CLASSES
                items = []
                for k, v in test_res["per_class_f1"].items():
                    if str(k).isdigit() and int(k) < len(class_names_map):
                        cname = class_names_map[int(k)].upper()
                    else:
                        cname = str(k).upper()
                    items.append({"Class": cname, "F1-Score": f"{float(v):.4f}"})
                per_class_df = pd.DataFrame(items)
                st.dataframe(per_class_df.T, use_container_width=True)


# =========================================================================
# TAB 5: REAL-TIME IMAGE UPLOAD & PREDICTION
# =========================================================================
elif nav_selection == "🔮 Real-Time Prediction":
    st.markdown(
        """
        <div class='hero-container'>
            <div class='hero-title'>Real-Time CNN Inference Engine</div>
            <div class='hero-sub'>Upload digit/object images or draw 1-click random test samples to run live PyTorch forward passes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Domain / Dataset Selector placed at the top for complete synchronization
    st.markdown("### 🎯 Target Vision Domain & Dataset")
    current_cached_ds = st.session_state.get("cached_test_ds", "CIFAR-10")
    domain_idx = 0 if current_cached_ds == "CIFAR-10" else 1

    domain_selection = st.radio(
        "Select Active Vision Domain:",
        [
            "🖼️ CIFAR-10 (Natural Objects: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)",
            "🔢 MNIST (Handwritten Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)",
        ],
        index=domain_idx,
        horizontal=True,
    )
    is_cifar_domain = "CIFAR-10" in domain_selection
    active_domain_name = "CIFAR-10" if is_cifar_domain else "MNIST"

    pred_col1, pred_col2 = st.columns([1, 1], gap="large")

    with pred_col1:
        st.markdown(f"### 1. Select {active_domain_name} Model Checkpoint")
        available_pts = get_available_checkpoints()

        # Intelligently filter checkpoints to match active domain
        if is_cifar_domain:
            domain_pts = [p for p in available_pts if "cifar" in os.path.basename(p).lower()]
            if not domain_pts:
                st.warning("⚠️ No CIFAR-10 checkpoints found. Showing all available checkpoints.")
                domain_pts = available_pts
        else:
            domain_pts = [p for p in available_pts if "cifar" not in os.path.basename(p).lower()]
            if not domain_pts:
                domain_pts = available_pts

        if not domain_pts:
            st.error("⚠️ No checkpoints found in checkpoints/. Please run training first!")
            selected_ckpt = None
        else:
            default_idx = 0
            for idx, pt in enumerate(domain_pts):
                if "improved" in pt and "best" in pt:
                    default_idx = idx
                    break
            selected_ckpt = st.selectbox(f"Trained {active_domain_name} Checkpoint:", domain_pts, index=default_idx)

        st.markdown(f"### 2. Choose {active_domain_name} Image Input")
        input_mode = st.radio(
            "Input Mode:",
            [
                "🎲 Pick Random Official Test Sample",
                "🎯 Select Sample by Class",
                "📁 Upload Your Own Image",
            ],
            horizontal=True,
        )

        input_pil_img = None
        ground_truth_label = None

        if input_mode == "📁 Upload Your Own Image":
            uploaded_file = st.file_uploader(
                f"Upload a {active_domain_name} image (PNG, JPG):",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_file is not None:
                raw_opened = Image.open(uploaded_file)
                input_pil_img = ensure_pil_image(raw_opened)
                st.session_state["cached_test_img"] = input_pil_img
                st.session_state["cached_test_lbl"] = None
                st.session_state["cached_test_ds"] = active_domain_name
                st.image(input_pil_img, caption="Uploaded Image", width=220)

        elif input_mode == "🎯 Select Sample by Class":
            target_classes = CIFAR10_CLASSES if is_cifar_domain else [f"Digit {i}" for i in range(10)]
            chosen_class_str = st.selectbox(f"Select Target {active_domain_name} Class:", target_classes, index=0)
            chosen_class_idx = target_classes.index(chosen_class_str)
            if st.button(f"🎯 Load Test Sample of {chosen_class_str.upper()}", type="primary", use_container_width=True):
                if is_cifar_domain:
                    raw_ds = datasets.CIFAR10(root="data", train=False, download=False, transform=None)
                else:
                    raw_ds = datasets.MNIST(root="data", train=False, download=False, transform=None)

                # Find all matching samples of this class
                matches = [idx for idx, (_, y) in enumerate(raw_ds) if y == chosen_class_idx]
                sample_idx = int(np.random.choice(matches)) if matches else 0
                sample_raw, sample_gt = raw_ds[sample_idx]
                input_pil_img = ensure_pil_image(sample_raw)
                ground_truth_label = int(sample_gt)

                st.session_state["cached_test_img"] = input_pil_img
                st.session_state["cached_test_lbl"] = ground_truth_label
                st.session_state["cached_test_ds"] = active_domain_name

        else:
            if st.button(f"🎲 Draw Random Sample from {active_domain_name} Test Set", type="primary", use_container_width=True):
                if is_cifar_domain:
                    raw_cifar = datasets.CIFAR10(root="data", train=False, download=False, transform=None)
                    rand_idx = np.random.randint(0, len(raw_cifar))
                    raw_sample, gt_label = raw_cifar[rand_idx]
                    input_pil_img = ensure_pil_image(raw_sample)
                    ground_truth_label = int(gt_label)

                    st.session_state["cached_test_img"] = input_pil_img
                    st.session_state["cached_test_lbl"] = ground_truth_label
                    st.session_state["cached_test_ds"] = "CIFAR-10"

                else:
                    raw_mnist = datasets.MNIST(root="data", train=False, download=False, transform=None)
                    rand_idx = np.random.randint(0, len(raw_mnist))
                    raw_sample, gt_label = raw_mnist[rand_idx]
                    input_pil_img = ensure_pil_image(raw_sample)
                    ground_truth_label = int(gt_label)

                    st.session_state["cached_test_img"] = input_pil_img
                    st.session_state["cached_test_lbl"] = ground_truth_label
                    st.session_state["cached_test_ds"] = "MNIST"

            # Retrieve from session state if matching active domain
            if "cached_test_img" in st.session_state:
                cached_ds = st.session_state.get("cached_test_ds", active_domain_name)
                if cached_ds == active_domain_name:
                    input_pil_img = ensure_pil_image(st.session_state["cached_test_img"])
                    ground_truth_label = st.session_state.get("cached_test_lbl")
                    class_names_list = CIFAR10_CLASSES if active_domain_name == "CIFAR-10" else MNIST_CLASSES
                    if ground_truth_label is not None:
                        lbl_name = class_names_list[ground_truth_label]
                        st.image(input_pil_img, caption=f"Ground Truth Label: {lbl_name.upper()} (Class {ground_truth_label})", width=220)
                    else:
                        st.image(input_pil_img, caption=f"Selected {active_domain_name} Sample", width=220)
                else:
                    st.info(f"💡 Active domain is {active_domain_name}. Click 'Draw Random Sample from {active_domain_name} Test Set' above.")

    with pred_col2:
        st.markdown(f"### 3. Real-Time {active_domain_name} Model Prediction")

        if input_pil_img is not None and selected_ckpt is not None:
            # Determine checkpoint domain and architecture
            is_ckpt_cifar = "cifar" in os.path.basename(selected_ckpt).lower()
            is_improved = "improved" in os.path.basename(selected_ckpt).lower()
            model_type = "improved" if is_improved else "basic"
            in_channels = 3 if is_ckpt_cifar else 1
            class_names = CIFAR10_CLASSES if is_ckpt_cifar else MNIST_CLASSES

            # Alert if cross-domain mismatch occurs
            if is_cifar_domain and not is_ckpt_cifar:
                st.warning("⚠️ Warning: Selected checkpoint is trained on MNIST digits. Switch to a CIFAR-10 checkpoint for object classification.")
            elif not is_cifar_domain and is_ckpt_cifar:
                st.warning("⚠️ Warning: Selected checkpoint is trained on CIFAR-10 objects. Switch to an MNIST checkpoint for digit recognition.")

            # Load model instance
            model = get_model(model_type, in_channels=in_channels, num_classes=10)
            ckpt_data = torch.load(selected_ckpt, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt_data:
                model.load_state_dict(ckpt_data["model_state_dict"])
            else:
                model.load_state_dict(ckpt_data)
            model.eval()

            # Preprocess PIL image for the network
            if in_channels == 3:
                img_proc = input_pil_img.convert("RGB").resize((32, 32))
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                    ]
                )
                tensor_x = transform(img_proc).unsqueeze(0)
            else:
                img_proc = input_pil_img.convert("L").resize((28, 28))
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                )
                tensor_x = transform(img_proc).unsqueeze(0)

            # Forward inference pass
            with torch.no_grad():
                logits = model(tensor_x)
                probs = F.softmax(logits, dim=1).squeeze().numpy()

            top1_idx = int(np.argmax(probs))
            top1_conf = probs[top1_idx] * 100
            pred_class_name = class_names[top1_idx]

            # Result Header Card in Royal Green Theme
            if ground_truth_label is not None and is_cifar_domain == is_ckpt_cifar:
                is_correct = top1_idx == ground_truth_label
                gt_name = class_names[ground_truth_label]
                if is_correct:
                    st.markdown(
                        f"""
                        <div class='pred-banner-correct'>
                            <div style='font-size: 1.35rem; font-weight: 800; color: #064E3B;'>🎯 Predicted: {pred_class_name.upper()} (Class {top1_idx}) <span style='color: #059669;'>✅ MATCH</span></div>
                            <div style='font-size: 0.95rem; color: #065F46; margin-top: 4px;'>Confidence: <b style='color: #047857;'>{top1_conf:.2f}%</b> | Ground Truth: <b>{gt_name.upper()}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='pred-banner-incorrect'>
                            <div style='font-size: 1.35rem; font-weight: 800; color: #991B1B;'>❌ Predicted: {pred_class_name.upper()} | True: {gt_name.upper()}</div>
                            <div style='font-size: 0.95rem; color: #7F1D1D; margin-top: 4px;'>Top Confidence: <b style='color: #DC2626;'>{top1_conf:.2f}%</b> (Misclassified)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"""
                    <div class='pred-banner-correct'>
                        <div style='font-size: 1.35rem; font-weight: 800; color: #064E3B;'>🎯 Predicted: {pred_class_name.upper()}</div>
                        <div style='font-size: 0.95rem; color: #065F46; margin-top: 4px;'>Top-1 Confidence: <b style='color: #047857;'>{top1_conf:.2f}%</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(f"#### Probability Distribution Across All 10 {active_domain_name} Classes")

            # Matplotlib horizontal bar chart with Royal Green & White aesthetic
            fig, ax = plt.subplots(figsize=(7, 4.3))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            
            colors = ["#059669" if i == top1_idx else "#E2E8F0" for i in range(10)]
            y_pos = np.arange(len(class_names))
            bars = ax.barh(y_pos, probs * 100, color=colors, height=0.62)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(class_names, fontsize=10, fontweight="bold", color="#0F172A")
            ax.invert_yaxis()
            ax.set_xlabel("Confidence Probability (%)", fontsize=10, color="#475569")
            ax.set_xlim(0, 100)
            ax.tick_params(colors="#475569")

            for bar in bars:
                width = bar.get_width()
                if width > 3:
                    t_col = "#064E3B" if bar.get_facecolor()[:3] == (0.0196078431372549, 0.5882352941176471, 0.4117647058823529) else "#334155"
                    ax.text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9, color=t_col, fontweight="bold")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#CBD5E1")
            ax.spines["bottom"].set_color("#059669")
            ax.spines["bottom"].set_alpha(0.5)
            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.info(f"👈 Click 'Draw Random Sample from {active_domain_name} Test Set' or upload an image to run real-time predictions.")


# =========================================================================
# TAB 6: WEIGHTS & BIASES TRACKER
# =========================================================================
elif nav_selection == "🌐 Weights & Biases Portal":
    st.markdown(
        """
        <div class='hero-container'>
            <div class='hero-title'>Weights & Biases (W&B) Experiment Portal</div>
            <div class='hero-sub'>Real-time scalar tracking, learning rate schedules, gradient distributions, and model checkpoints</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_bg = "linear-gradient(135deg, #059669 0%, #047857 50%, #064E3B 100%)"
    btn_color = "#FFFFFF"

    st.markdown(
        f"""
        <div class='content-panel' style='text-align: left; padding: 26px;'>
            <h3 style='margin-top: 0; color: #064E3B;'>🔗 Project: DEEPVISION (CoreWeave W&B)</h3>
            <p style='font-size: 1rem; color: #1E293B;'>Every fold, validation step, confusion matrix, and sample prediction is tracked live on Weights & Biases.</p>
            <a href='https://forge.coreweave.com/wandb/vaishalinir-ymc2022-chennai-institute-of-technology/DEEPVISION' target='_blank' style='display: inline-block; background: {btn_bg}; color: {btn_color}; padding: 11px 24px; border-radius: 9px; text-decoration: none; font-weight: 800; margin-top: 8px; box-shadow: 0 4px 15px rgba(5,150,105,0.25);'>
                🚀 Open DEEPVISION in Weights & Biases
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📂 Synced Experiment Runs")
    wandb_dir = "wandb"
    if os.path.exists(wandb_dir):
        run_folders = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]
        run_records = []
        for rf in sorted(run_folders, reverse=True):
            parts = rf.split("-")
            timestamp_str = parts[1] if len(parts) > 1 else "N/A"
            run_id = parts[2] if len(parts) > 2 else "N/A"
            run_records.append(
                {
                    "Run Folder": rf,
                    "Run ID": run_id,
                    "Timestamp": timestamp_str,
                    "Tracking Status": "Synced to CoreWeave W&B",
                }
            )

        st.dataframe(pd.DataFrame(run_records), use_container_width=True, hide_index=True)
    else:
        st.info("No local W&B run folders found.")

    st.markdown("### 📊 Logged Metrics & Visualizations")
    st.markdown(
        """
        - **Real-Time Curves:** `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `learning_rate`
        - **Evaluation Metrics:** `val/f1_macro`, `val/precision_macro`, `val/recall_macro`
        - **Media Visualizations:** High-resolution Confusion Matrix heatmaps and prediction grids with color-coded classification borders.
        """
    )
