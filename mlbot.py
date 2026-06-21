"""
Telegram Scam Detection System - ML/DL Implementation (ENHANCED)
+ OCR for images + Speech-to-Text for audio + LightGBM instead of Extra Trees
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# Telegram
from telethon import TelegramClient, events
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Database
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    VotingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

# OCR
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR not available. Install with: pip install pillow pytesseract")

# Speech-to-Text
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    logging.warning("Speech recognition not available. Install with: pip install SpeechRecognition pydub")

# NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    logger.warning("NLTK downloads failed")

# ==================== CONFIGURATION ====================
class Config:
    # Telegram
    TELEGRAM_API_ID = "xxxx"  # Replace with your API ID
    TELEGRAM_API_HASH = "xxxx"  # Replace with your API Hash
    TELEGRAM_PHONE = "xxxxx"  # Replace with your phone number
    TELEGRAM_SESSION_NAME = "scam_detector_session"
    # Dataset
    DATASET_PATH = r"C:\Users\SATHVIK\TGSRTC\scam_final.csv"
    
    # Directories
    MODELS_DIR = "trained_models"
    PLOTS_DIR = "model_plots"
    TEMP_DIR = "temp_media"
    
    # MongoDB
    MONGODB_URI = "xvxvhsvdajsvdadkadkas"
    DATABASE_NAME = "telegram_scam_detection_ml"
    
    # Email
    SMTP_HOST = "smtp.gmail.com"
    SMTP_PORT = 587
    SMTP_USER = "nadjdjasklaslfaslb"
    SMTP_PASS = "asgddakbbfaskjhdbasdbaskldbl"
    CYBER_CELL_EMAIL = "eppakayalasathvik72@gmail.com"
    
    # ML Parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_FEATURES = 5000
    
    # Risk Thresholds
    GROUP_RISK_THRESHOLD = 10

os.makedirs(Config.TEMP_DIR, exist_ok=True)


# ==================== MEDIA PROCESSOR ====================
class MediaProcessor:
    """Process images and audio files"""
    
    @staticmethod
    async def extract_text_from_image(image_path: str) -> str:
        """Extract text from image using OCR"""
        if not OCR_AVAILABLE:
            logger.warning("OCR not available")
            return ""
        
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            logger.info(f"✅ OCR extracted {len(text)} characters")
            return text.strip()
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    @staticmethod
    async def convert_audio_to_text(audio_path: str) -> str:
        """Convert audio to text using speech recognition"""
        if not STT_AVAILABLE:
            logger.warning("Speech recognition not available")
            return ""
        
        try:
            # Convert to WAV if needed
            audio_ext = os.path.splitext(audio_path)[1].lower()
            wav_path = audio_path
            
            if audio_ext != '.wav':
                logger.info(f"Converting {audio_ext} to WAV...")
                audio = AudioSegment.from_file(audio_path)
                wav_path = audio_path.replace(audio_ext, '.wav')
                audio.export(wav_path, format="wav")
            
            # Speech recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                logger.info(f"✅ Speech-to-Text: {len(text)} characters")
                return text
        
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return ""
    
    @staticmethod
    async def download_telegram_media(client, message) -> Optional[str]:
        """Download media from Telegram message"""
        try:
            if not message.media:
                return None
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            media_type = None
            
            if isinstance(message.media, MessageMediaPhoto):
                media_type = "photo"
                file_ext = ".jpg"
            elif isinstance(message.media, MessageMediaDocument):
                mime = message.media.document.mime_type
                if 'image' in mime:
                    media_type = "image"
                    file_ext = ".jpg"
                elif 'audio' in mime or 'ogg' in mime:
                    media_type = "audio"
                    file_ext = ".ogg"
                else:
                    return None
            else:
                return None
            
            if not media_type:
                return None
            
            filepath = os.path.join(Config.TEMP_DIR, f"{media_type}_{timestamp}{file_ext}")
            await client.download_media(message, filepath)
            logger.info(f"✅ Downloaded {media_type}: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Media download error: {e}")
            return None


# ==================== TEXT PREPROCESSOR ====================
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str) or len(text) == 0:
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        try:
            tokens = word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(word) 
                for word in tokens 
                if word not in self.stop_words and len(word) > 2
            ]
            return ' '.join(tokens)
        except:
            return text


# ==================== MODEL TRAINER ====================
class ModelTrainer:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.vectorizers = {}
        self.metrics = {}
        self.best_params = {}
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load dataset with Parquet conversion and proper cleaning"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        parquet_path = self.dataset_path.replace('.csv', '.parquet')
        
        if os.path.exists(parquet_path):
            logger.info(f"✅ Loading from Parquet (fast): {parquet_path}")
            df = pd.read_parquet(parquet_path)
            logger.info(f"Loaded {len(df)} rows from Parquet")
        
        else:
            logger.info("📊 Converting CSV to Parquet (one-time process)...")
            
            chunk_size = 10000
            chunks = []
            
            logger.info("Reading CSV in chunks...")
            try:
                for i, chunk in enumerate(pd.read_csv(
                    self.dataset_path,
                    chunksize=chunk_size,
                    low_memory=False,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    engine='python'
                )):
                    chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed', case=False, na=False)]
                    chunks.append(chunk)
                    logger.info(f"  Chunk {i+1}: {len(chunk)} rows")
                
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"✅ Total loaded: {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Chunk reading failed: {e}")
                logger.info("Trying direct read...")
                df = pd.read_csv(self.dataset_path, encoding='utf-8', on_bad_lines='skip')
            
            df = self.clean_dataframe(df)
            
            try:
                logger.info(f"💾 Saving as Parquet: {parquet_path}")
                df.to_parquet(parquet_path, index=False, compression='snappy')
                logger.info("✅ Parquet saved! Future loads will be much faster.")
            except Exception as e:
                logger.warning(f"Could not save Parquet: {e}")
        
        logger.info(f"\nDataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        if 'message' not in df.columns or 'classification' not in df.columns:
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        
        logger.info(f"\n📊 Class distribution:")
        print(df['classification'].value_counts())
        
        logger.info("\n🔄 Preprocessing text (this may take a while)...")
        
        batch_size = 5000
        cleaned_texts = []
        
        for i in range(0, len(df), batch_size):
            batch = df['message'].iloc[i:i+batch_size]
            cleaned_batch = batch.apply(self.preprocessor.clean_text)
            cleaned_texts.extend(cleaned_batch.tolist())
            progress = min(i + batch_size, len(df))
            logger.info(f"  Processed {progress}/{len(df)} messages ({progress/len(df)*100:.1f}%)")
        
        df['cleaned_text'] = cleaned_texts
        df = df[df['cleaned_text'].str.len() > 0]
        logger.info(f"After cleaning: {len(df)} rows")
        
        df['label'] = self.label_encoder.fit_transform(df['classification'])
        self.class_names = self.label_encoder.classes_.tolist()
        
        logger.info(f"✅ Classes: {self.class_names}")
        
        with open(os.path.join(Config.MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        return df, df['cleaned_text'].values, df['label'].values
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe and fix corrupted labels"""
        logger.info("🧹 Cleaning dataframe...")
        
        df.columns = df.columns.str.lower().str.strip()
        
        message_col = None
        class_col = None
        
        for col in df.columns:
            col_str = str(col).lower()
            if any(kw in col_str for kw in ['message', 'text', 'content', 'email', 'body']):
                message_col = col
            elif any(kw in col_str for kw in ['classification', 'label', 'class', 'category', 'type']):
                class_col = col
        
        if not message_col or not class_col:
            raise ValueError(f"Could not find required columns. Found: {df.columns.tolist()}")
        
        df = df.rename(columns={message_col: 'message', class_col: 'classification'})
        df = df[['message', 'classification']]
        df = df.dropna()
        
        df['message'] = df['message'].astype(str).str.strip()
        df['classification'] = df['classification'].astype(str).str.strip()
        
        logger.info("🔧 Fixing classification labels...")
        
        def normalize_label(label):
            label_str = str(label).lower().strip()
            
            if any(kw in label_str for kw in ['normal', 'legitimate', 'ham', 'safe']):
                return 'Normal'
            elif any(kw in label_str for kw in ['scam', 'fraud', 'phishing', 'spam']):
                return 'Scam'
            elif any(kw in label_str for kw in ['suspicious', 'questionable']):
                return 'Suspicious'
            
            if len(label_str) > 50:
                return None
            
            return 'Normal'
        
        df['classification'] = df['classification'].apply(normalize_label)
        
        before = len(df)
        df = df[df['classification'].notna()]
        after = len(df)
        
        if before > after:
            logger.warning(f"Removed {before - after} rows with corrupted labels")
        
        df = df[df['message'].str.len().between(10, 5000)]
        
        class_counts = df['classification'].value_counts()
        logger.info(f"\n✅ Cleaned class distribution:\n{class_counts}")
        
        if class_counts.min() < 2:
            raise ValueError(f"Some classes have < 2 samples: {class_counts}")
        
        return df
    
    def train_all_models(self):
        """Train all ML models including LightGBM"""
        logger.info("="*80)
        logger.info("🚀 STARTING MODEL TRAINING PIPELINE")
        logger.info("="*80)
        
        df, X, y = self.load_and_preprocess_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"\n📊 Training set: {len(X_train)} samples")
        logger.info(f"📊 Test set: {len(X_test)} samples")
        
        logger.info("\n" + "="*80)
        logger.info("📝 CREATING TF-IDF FEATURES")
        logger.info("="*80)
        
        tfidf = TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        self.vectorizers['tfidf'] = tfidf
        with open(os.path.join(Config.MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(tfidf, f)
        
        logger.info(f"✅ Feature matrix: {X_train_tfidf.shape}")
        
        self.train_individual_models(X_train_tfidf, X_test_tfidf, y_train, y_test)
        self.create_ensemble_model(X_train_tfidf, X_test_tfidf, y_train, y_test)
        self.generate_all_visualizations(X_test_tfidf, y_test)
        self.save_metrics()
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
    
    def train_individual_models(self, X_train, X_test, y_train, y_test):
        """Train individual models with LightGBM instead of Extra Trees"""
        
        # 1. Naive Bayes
        logger.info("\n[1/7] Training Naive Bayes...")
        nb = MultinomialNB(alpha=1.0)
        nb.fit(X_train, y_train)
        self.models['naive_bayes'] = nb
        self.evaluate_model('Naive Bayes', nb, X_test, y_test)
        
        # 2. Logistic Regression
        logger.info("\n[2/7] Training Logistic Regression...")
        lr = LogisticRegression(C=10.0, max_iter=500, random_state=Config.RANDOM_STATE)
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        self.evaluate_model('Logistic Regression', lr, X_test, y_test)
        
        # 3. Random Forest
        logger.info("\n[3/7] Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=Config.RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        self.evaluate_model('Random Forest', rf, X_test, y_test)
        
        # 4. Gradient Boosting
        logger.info("\n[4/7] Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=Config.RANDOM_STATE)
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        self.evaluate_model('Gradient Boosting', gb, X_test, y_test)
        
        # 5. SVM
        logger.info("\n[5/7] Training SVM...")
        svm = SVC(C=10.0, kernel='rbf', probability=True, random_state=Config.RANDOM_STATE)
        svm.fit(X_train, y_train)
        self.models['svm'] = svm
        self.evaluate_model('SVM', svm, X_test, y_test)
        
        # 6. LightGBM (replaces Extra Trees)
        logger.info("\n[6/7] Training LightGBM...")
        if LIGHTGBM_AVAILABLE:
            lgbm = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=20,
                num_leaves=31,
                random_state=Config.RANDOM_STATE,
                verbose=-1
            )
            lgbm.fit(X_train, y_train)
            self.models['lightgbm'] = lgbm
            self.evaluate_model('LightGBM', lgbm, X_test, y_test)
        else:
            logger.warning("⚠️ LightGBM not available. Install with: pip install lightgbm")
        
        # 7. AdaBoost
        logger.info("\n[7/7] Training AdaBoost...")
        ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=Config.RANDOM_STATE)
        ada.fit(X_train, y_train)
        self.models['adaboost'] = ada
        self.evaluate_model('AdaBoost', ada, X_test, y_test)
        
        # Save models
        for name, model in self.models.items():
            if name != 'ensemble':
                with open(os.path.join(Config.MODELS_DIR, f'{name}_model.pkl'), 'wb') as f:
                    pickle.dump(model, f)
    
    def create_ensemble_model(self, X_train, X_test, y_train, y_test):
        """Create ensemble model"""
        logger.info("\n" + "="*80)
        logger.info("🎯 CREATING ENSEMBLE MODEL")
        logger.info("="*80)
        
        sorted_models = sorted(
            [(name, self.metrics[name]['f1_score']) for name in self.metrics.keys()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        logger.info("\nTop 5 models:")
        for name, score in sorted_models:
            logger.info(f"  ✓ {name}: F1={score:.4f}")
        
        estimators = [(name, self.models[name]) for name, _ in sorted_models]
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        ensemble.fit(X_train, y_train)
        
        self.models['ensemble'] = ensemble
        self.evaluate_model('Ensemble', ensemble, X_test, y_test)
        
        with open(os.path.join(Config.MODELS_DIR, 'ensemble_model.pkl'), 'wb') as f:
            pickle.dump(ensemble, f)
    
    def evaluate_model(self, name: str, model, X_test, y_test):
        """Evaluate model"""
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        self.metrics[name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"\n✅ {name}:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        
        self.plot_confusion_matrix(cm, name)
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'), dpi=150)
        plt.close()
    
    def generate_all_visualizations(self, X_test, y_test):
        """Generate all visualizations"""
        logger.info("\n📊 Generating visualizations...")
        self.plot_model_comparison()
        self.plot_per_class_performance()
        self.plot_feature_importance()
    
    def plot_model_comparison(self):
        """Plot model comparison"""
        model_names = list(self.metrics.keys())
        metrics_data = {
            'accuracy': [self.metrics[m]['accuracy'] for m in model_names],
            'precision': [self.metrics[m]['precision'] for m in model_names],
            'recall': [self.metrics[m]['recall'] for m in model_names],
            'f1_score': [self.metrics[m]['f1_score'] for m in model_names]
        }
        
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=18, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, 'model_comparison.png'), dpi=150)
        plt.close()
    
    def plot_per_class_performance(self):
        """Plot per-class performance"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(self.metrics.items()):
            if idx >= 8:
                break
            
            report = metrics['classification_report']
            f1_scores = [report[cls]['f1-score'] for cls in self.class_names]
            
            ax = axes[idx]
            bars = ax.bar(self.class_names, f1_scores, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('F1 Score')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for idx in range(len(self.metrics), 8):
            axes[idx].axis('off')
        
        plt.suptitle('Per-Class F1 Scores', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, 'per_class_performance.png'), dpi=150)
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        tree_models = ['random_forest', 'gradient_boosting', 'lightgbm']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, model_name in enumerate(tree_models):
            if model_name not in self.models:
                axes[idx].text(0.5, 0.5, f'{model_name} not available', ha='center', va='center')
                axes[idx].set_title(model_name.replace('_', ' ').title())
                continue
            
            model = self.models[model_name]
            if not hasattr(model, 'feature_importances_'):
                continue
            
            feature_names = self.vectorizers['tfidf'].get_feature_names_out()
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]
            
            ax = axes[idx]
            ax.barh(range(20), importances[indices], color='teal', alpha=0.7)
            ax.set_yticks(range(20))
            ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Top 20 Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, 'feature_importance.png'), dpi=150)
        plt.close()
    
    def save_metrics(self):
        """Save metrics to JSON"""
        metrics_data = {
            'metrics': self.metrics,
            'best_params': self.best_params,
            'label_classes': self.class_names,
            'training_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(Config.MODELS_DIR, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"\n✅ Metrics saved to {Config.MODELS_DIR}/metrics.json")


# ==================== SCAM PREDICTOR ====================
class ScamPredictor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            with open(os.path.join(Config.MODELS_DIR, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open(os.path.join(Config.MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            model_files = [
                'naive_bayes', 'logistic_regression', 'random_forest',
                'gradient_boosting', 'svm', 'lightgbm', 'adaboost', 'ensemble'
            ]
            
            for model_name in model_files:
                model_path = os.path.join(Config.MODELS_DIR, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            logger.info(f"✅ Loaded {len(self.models)} models")
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict classification"""
        cleaned_text = self.preprocessor.clean_text(text)
        
        if not cleaned_text:
            return {
                'final_classification': 'Normal',
                'confidence': 0.0,
                'ensemble_prediction': 'Normal',
                'ensemble_confidence': 0.0,
                'individual_predictions': {},
                'individual_confidences': {},
                'reasoning': 'Empty text'
            }
        
        X = self.vectorizer.transform([cleaned_text])
        
        predictions = {}
        confidences = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                pred_label = self.label_encoder.inverse_transform([pred])[0]
                predictions[name] = pred_label
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    probabilities[name] = proba.tolist()
                    confidences[name] = float(max(proba) * 100)
                else:
                    confidences[name] = 50.0
            
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
                predictions[name] = 'Normal'
                confidences[name] = 0.0
        
        if 'ensemble' in predictions:
            final_prediction = predictions['ensemble']
            final_confidence = confidences.get('ensemble', 50.0)
        else:
            vote_counts = Counter(predictions.values())
            final_prediction = vote_counts.most_common(1)[0][0]
            agreeing_confidences = [
                conf for model, conf in confidences.items()
                if predictions[model] == final_prediction
            ]
            final_confidence = np.mean(agreeing_confidences) if agreeing_confidences else 50.0
        
        reasoning = self.generate_reasoning(predictions, confidences, final_prediction)
        
        return {
            'final_classification': final_prediction,
            'confidence': float(final_confidence),
            'ensemble_prediction': predictions.get('ensemble', final_prediction),
            'ensemble_confidence': confidences.get('ensemble', final_confidence),
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'individual_probabilities': probabilities,
            'reasoning': reasoning,
            'voting_summary': dict(Counter(predictions.values()))
        }
    
    def generate_reasoning(self, predictions, confidences, final_pred):
        """Generate reasoning"""
        total = len(predictions)
        agreeing = sum(1 for p in predictions.values() if p == final_pred)
        
        reasoning = f"{agreeing}/{total} models classified as '{final_pred}'. "
        
        if agreeing == total:
            reasoning += "Unanimous agreement."
        elif agreeing > total * 0.7:
            reasoning += "Strong consensus."
        elif agreeing > total * 0.5:
            reasoning += "Majority agreement."
        else:
            reasoning += "Mixed predictions."
        
        high_conf = [name for name, conf in confidences.items() if conf > 80 and predictions[name] == final_pred]
        if high_conf:
            reasoning += f" High confidence from: {', '.join(high_conf)}."
        
        return reasoning


# ==================== DATABASE MANAGER ====================
class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
    
    async def connect(self):
        self.client = AsyncIOMotorClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        
        await self.db.messages.create_index([("group_id", ASCENDING), ("timestamp", DESCENDING)])
        await self.db.users.create_index([("user_id", ASCENDING)])
        await self.db.groups.create_index([("group_id", ASCENDING)])
        await self.db.reports.create_index([("group_id", ASCENDING), ("timestamp", DESCENDING)])
        
        logger.info("✅ Connected to MongoDB")
    
    async def close(self):
        if self.client:
            self.client.close()
    
    async def save_message(self, message_data: Dict[str, Any]):
        await self.db.messages.insert_one(message_data)
    
    async def update_user_risk(self, user_id: int, increment: int = 1):
        result = await self.db.users.find_one_and_update(
            {"user_id": user_id},
            {"$inc": {"risk_score": increment}, "$set": {"last_updated": datetime.now(timezone.utc)}},
            upsert=True,
            return_document=True
        )
        return result
    
    async def update_group_risk(self, group_id: int, increment: int = 1):
        result = await self.db.groups.find_one_and_update(
            {"group_id": group_id},
            {"$inc": {"risk_score": increment, "total_messages": 1}, "$set": {"last_updated": datetime.now(timezone.utc)}},
            upsert=True,
            return_document=True
        )
        return result
    
    async def save_user_info(self, user_data: Dict[str, Any]):
        await self.db.users.update_one({"user_id": user_data["user_id"]}, {"$set": user_data}, upsert=True)
    
    async def save_group_info(self, group_data: Dict[str, Any]):
        await self.db.groups.update_one({"group_id": group_data["group_id"]}, {"$set": group_data}, upsert=True)
    
    async def get_group_suspicious_users(self, group_id: int):
        messages = self.db.messages.find({
            "group_id": group_id,
            "ml_prediction.final_classification": {"$in": ["Suspicious", "Scam"]}
        })
        
        user_ids = set()
        async for msg in messages:
            user_ids.add(msg["user_id"])
        
        users = []
        for user_id in user_ids:
            user = await self.db.users.find_one({"user_id": user_id})
            if user:
                users.append(user)
        
        return users
    
    async def get_group_stats(self, group_id: int):
        group = await self.db.groups.find_one({"group_id": group_id})
        
        total_messages = await self.db.messages.count_documents({"group_id": group_id})
        suspicious_messages = await self.db.messages.count_documents({
            "group_id": group_id,
            "ml_prediction.final_classification": "Suspicious"
        })
        fraud_messages = await self.db.messages.count_documents({
            "group_id": group_id,
            "ml_prediction.final_classification": "Scam"
        })
        
        return {
            "group": group,
            "total_messages": total_messages,
            "suspicious_messages": suspicious_messages,
            "fraud_messages": fraud_messages
        }
    
    async def save_report(self, report_data: Dict[str, Any]):
        await self.db.reports.insert_one(report_data)


# ==================== EMAIL REPORTER ====================
class EmailReporter:
    @staticmethod
    async def send_fraud_report(group_id: int, db: DatabaseManager):
        """Send fraud alert"""
        stats = await db.get_group_stats(group_id)
        group = stats["group"]
        suspicious_users = await db.get_group_suspicious_users(group_id)
        
        excel_buffer = EmailReporter.create_suspicious_users_excel(suspicious_users)
        
        msg = MIMEMultipart()
        msg['From'] = Config.SMTP_USER
        msg['To'] = Config.CYBER_CELL_EMAIL
        msg['Subject'] = f"🚨 FRAUD ALERT: {group.get('title', 'Unknown')}"
        
        body = f"""
TELEGRAM SCAM DETECTION - ML FRAUD ALERT

Group: {group.get('title', 'N/A')}
Risk Score: {group.get('risk_score', 0)}/10
Messages: {stats['total_messages']}
Suspicious: {stats['suspicious_messages']}
Fraud: {stats['fraud_messages']}
Suspicious Users: {len(suspicious_users)}

Report: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(excel_buffer.getvalue())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename=users_{group_id}.xlsx')
        msg.attach(part)
        
        try:
            server = smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT)
            server.starttls()
            server.login(Config.SMTP_USER, Config.SMTP_PASS)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"✅ Report sent for group {group_id}")
            
            await db.save_report({
                "group_id": group_id,
                "timestamp": datetime.now(timezone.utc),
                "suspicious_users_count": len(suspicious_users),
                "total_messages": stats['total_messages'],
                "risk_score": group.get('risk_score', 0)
            })
        
        except Exception as e:
            logger.error(f"❌ Email error: {e}")
    
    @staticmethod
    def create_suspicious_users_excel(users: List[Dict[str, Any]]) -> io.BytesIO:
        """Create Excel"""
        data = []
        for user in users:
            data.append({
                "User ID": user.get("user_id"),
                "Username": user.get("username", "N/A"),
                "First Name": user.get("first_name", "N/A"),
                "Last Name": user.get("last_name", "N/A"),
                "Risk Score": user.get("risk_score", 0)
            })
        
        df = pd.DataFrame(data)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Suspicious Users')
        buffer.seek(0)
        return buffer


# ==================== TELEGRAM MONITOR ====================
class TelegramMonitor:
    def __init__(self):
        self.client = None
        self.db = DatabaseManager()
        self.predictor = ScamPredictor()
        self.media_processor = MediaProcessor()
    
    async def initialize(self):
        await self.db.connect()
        
        self.client = TelegramClient(Config.TELEGRAM_SESSION_NAME, Config.TELEGRAM_API_ID, Config.TELEGRAM_API_HASH)
        await self.client.start(phone=Config.TELEGRAM_PHONE)
        logger.info("✅ Telegram client started")
        
        self.client.add_event_handler(self.handle_message, events.NewMessage())
    
    async def handle_message(self, event):
        try:
            message = event.message
            
            if not message.is_group:
                return
            
            chat = await event.get_chat()
            sender = await event.get_sender()
            
            group_data = {
                "group_id": chat.id,
                "title": getattr(chat, 'title', 'Unknown'),
                "username": getattr(chat, 'username', None),
                "member_count": getattr(chat, 'participants_count', None),
                "last_updated": datetime.now(timezone.utc)
            }
            await self.db.save_group_info(group_data)
            
            user_data = {
                "user_id": sender.id,
                "username": getattr(sender, 'username', None),
                "first_name": getattr(sender, 'first_name', ''),
                "last_name": getattr(sender, 'last_name', ''),
                "is_bot": getattr(sender, 'bot', False),
                "last_seen": datetime.now(timezone.utc)
            }
            await self.db.save_user_info(user_data)
            
            # Extract content from text, images, or audio
            content = await self.extract_message_content(message)
            
            if not content or len(content.strip()) == 0:
                content = "[No text content]"
            
            prediction = self.predictor.predict(content)
            
            message_data = {
                "message_id": message.id,
                "group_id": chat.id,
                "user_id": sender.id,
                "content": content,
                "has_media": message.media is not None,
                "media_type": self.get_media_type(message),
                "timestamp": message.date,
                "ml_prediction": prediction,
                "processed_at": datetime.now(timezone.utc)
            }
            await self.db.save_message(message_data)
            
            classification = prediction['final_classification']
            
            if classification in ["Suspicious", "Scam"]:
                await self.db.update_user_risk(sender.id, 1)
                group_result = await self.db.update_group_risk(chat.id, 1)
                
                logger.warning(f"⚠️ {classification} in {group_data['title']}")
                logger.warning(f"Content preview: {content[:100]}")
                logger.warning(f"Confidence: {prediction['confidence']:.2f}%")
                
                if group_result.get('risk_score', 0) >= Config.GROUP_RISK_THRESHOLD:
                    logger.critical(f"🚨 Group {chat.id} threshold reached!")
                    await EmailReporter.send_fraud_report(chat.id, self.db)
                    await self.db.db.groups.update_one(
                        {"group_id": chat.id},
                        {"$set": {"risk_score": 0, "last_reported": datetime.now(timezone.utc)}}
                    )
        
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
    async def extract_message_content(self, message) -> str:
        """Extract text content from message (text, image OCR, or audio STT)"""
        content_parts = []
        
        # Get text content
        if message.text:
            content_parts.append(message.text)
        
        # Process media if available
        if message.media:
            try:
                media_path = await self.media_processor.download_telegram_media(self.client, message)
                
                if media_path:
                    # Extract text based on media type
                    if 'photo' in media_path or 'image' in media_path:
                        logger.info("📷 Processing image with OCR...")
                        ocr_text = await self.media_processor.extract_text_from_image(media_path)
                        if ocr_text:
                            content_parts.append(f"[Image Text: {ocr_text}]")
                            logger.info(f"✅ OCR extracted: {ocr_text[:100]}")
                    
                    elif 'audio' in media_path or 'voice' in media_path:
                        logger.info("🎤 Processing audio with Speech-to-Text...")
                        stt_text = await self.media_processor.convert_audio_to_text(media_path)
                        if stt_text:
                            content_parts.append(f"[Audio Text: {stt_text}]")
                            logger.info(f"✅ STT extracted: {stt_text[:100]}")
                    
                    # Clean up temp file
                    try:
                        os.remove(media_path)
                    except:
                        pass
            
            except Exception as e:
                logger.error(f"Media processing error: {e}")
        
        return " ".join(content_parts) if content_parts else ""
    
    def get_media_type(self, message) -> Optional[str]:
        """Get media type from message"""
        if not message.media:
            return None
        
        if isinstance(message.media, MessageMediaPhoto):
            return "photo"
        elif isinstance(message.media, MessageMediaDocument):
            mime = message.media.document.mime_type
            if 'image' in mime:
                return "image"
            elif 'audio' in mime or 'ogg' in mime:
                return "audio"
            elif 'video' in mime:
                return "video"
            else:
                return "document"
        return "other"
    
    async def run(self):
        logger.info("🚀 Monitor running with OCR and Speech-to-Text...")
        await self.client.run_until_disconnected()
    
    async def close(self):
        if self.client:
            await self.client.disconnect()
        await self.db.close()


# ==================== FASTAPI ====================
app = FastAPI(title="ML Scam Detection API (Enhanced)", version="3.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

monitor = None
predictor = None

@app.on_event("startup")
async def startup_event():
    global monitor, predictor
    try:
        predictor = ScamPredictor()
        monitor = TelegramMonitor()
        await monitor.initialize()
        asyncio.create_task(monitor.run())
        logger.info("✅ Server started with OCR and STT")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global monitor
    if monitor:
        await monitor.close()

@app.get("/")
async def root():
    return {
        "status": "online", 
        "service": "ML Scam Detection (Enhanced)", 
        "version": "3.0",
        "models": len(predictor.models) if predictor else 0,
        "features": ["OCR", "Speech-to-Text", "LightGBM"]
    }

@app.get("/models/metrics")
async def get_model_metrics():
    try:
        with open(os.path.join(Config.MODELS_DIR, 'metrics.json'), 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/plots/{plot_name}")
async def get_plot(plot_name: str):
    plot_path = os.path.join(Config.PLOTS_DIR, plot_name)
    if os.path.exists(plot_path):
        return FileResponse(plot_path)
    raise HTTPException(status_code=404, detail="Plot not found")

@app.get("/models/plots")
async def list_plots():
    try:
        plots = []
        if os.path.exists(Config.PLOTS_DIR):
            for file in os.listdir(Config.PLOTS_DIR):
                if file.endswith('.png'):
                    plots.append(file)
        return {"plots": plots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_text(request: dict):
    if not predictor:
        raise HTTPException(status_code=503, detail="Not initialized")
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text required")
    return predictor.predict(text)

@app.get("/messages")
async def get_messages(limit: int = 1000, skip: int = 0):
    if not monitor:
        raise HTTPException(status_code=503, detail="Not initialized")
    try:
        messages = []
        async for msg in monitor.db.db.messages.find().sort("timestamp", DESCENDING).skip(skip).limit(limit):
            msg["_id"] = str(msg["_id"])
            if isinstance(msg.get("timestamp"), datetime):
                msg["timestamp"] = msg["timestamp"].isoformat()
            if isinstance(msg.get("processed_at"), datetime):
                msg["processed_at"] = msg["processed_at"].isoformat()
            messages.append(msg)
        return {"messages": messages, "total": len(messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/groups")
async def get_groups():
    if not monitor:
        raise HTTPException(status_code=503, detail="Not initialized")
    try:
        groups = []
        async for group in monitor.db.db.groups.find():
            group["_id"] = str(group["_id"])
            if isinstance(group.get("last_updated"), datetime):
                group["last_updated"] = group["last_updated"].isoformat()
            if isinstance(group.get("last_reported"), datetime):
                group["last_reported"] = group["last_reported"].isoformat()
            groups.append(group)
        return {"groups": groups, "total": len(groups)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    if not monitor:
        raise HTTPException(status_code=503, detail="Not initialized")
    try:
        total = await monitor.db.db.messages.count_documents({})
        fraud = await monitor.db.db.messages.count_documents({"ml_prediction.final_classification": "Scam"})
        suspicious = await monitor.db.db.messages.count_documents({"ml_prediction.final_classification": "Suspicious"})
        normal = await monitor.db.db.messages.count_documents({"ml_prediction.final_classification": "Normal"})
        groups = await monitor.db.db.groups.count_documents({})
        
        return {
            "total_messages": total,
            "fraud_messages": fraud,
            "suspicious_messages": suspicious,
            "normal_messages": normal,
            "total_groups": groups,
            "fraud_percentage": round((fraud / total * 100), 2) if total > 0 else 0,
            "suspicious_percentage": round((suspicious / total * 100), 2) if total > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class JoinGroupRequest(BaseModel):
    group_link: str

@app.post("/join-group")
async def join_group(request: JoinGroupRequest):
    if not monitor:
        raise HTTPException(status_code=503, detail="Not initialized")
    try:
        result = await monitor.client(JoinChannelRequest(request.group_link))
        chat = result.chats[0] if result.chats else None
        if chat:
            return {"status": "success", "message": f"Joined: {chat.title}", "group_id": chat.id}
        else:
            raise HTTPException(status_code=400, detail="Could not join")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== MAIN ====================
if __name__ == "__main__":
    import sys
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   TELEGRAM SCAM DETECTION - ML/DL (ENHANCED)            ║
    ║   OCR + Speech-to-Text + LightGBM + Parquet Support     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("\n📚 TRAINING MODE")
        print("="*80)
        
        if not os.path.exists(Config.DATASET_PATH):
            print(f"\n❌ Dataset not found: {Config.DATASET_PATH}")
            sys.exit(1)
        
        trainer = ModelTrainer(Config.DATASET_PATH)
        trainer.train_all_models()
        
        print("\n✅ Training complete!")
        print("   python ml_scam_detector_enhanced.py")
    
    else:
        print("\n🚀 SERVER MODE")
        print("="*80)
        
        if not os.path.exists(Config.MODELS_DIR):
            print("\n⚠️  Train first:")
            print("   python ml_scam_detector_enhanced.py train")
            sys.exit(1)
        
        print("\nModels:")
        for file in os.listdir(Config.MODELS_DIR):
            print(f"  ✓ {file}")
        
        print("\nFeatures:")
        print(f"  {'✅' if OCR_AVAILABLE else '❌'} OCR (Images)")
        print(f"  {'✅' if STT_AVAILABLE else '❌'} Speech-to-Text (Audio)")
        print(f"  {'✅' if LIGHTGBM_AVAILABLE else '❌'} LightGBM Model")
        
        if not OCR_AVAILABLE:
            print("\n⚠️  Install OCR: pip install pillow pytesseract")
        if not STT_AVAILABLE:
            print("⚠️  Install STT: pip install SpeechRecognition pydub")
        if not LIGHTGBM_AVAILABLE:
            print("⚠️  Install LightGBM: pip install lightgbm")
        
        print("\nStarting server...")
        print("API: http://localhost:8000")
        print("Dashboard: ml_dashboard.html")
        
        uvicorn.run(app, host="0.0.0.0", port=8001)
