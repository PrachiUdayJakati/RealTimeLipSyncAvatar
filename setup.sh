#!/bin/bash

# Real-Time Lipsync Avatar Setup Script
# Optimized for 16GB GPU VRAM + 54GB RAM

echo "🚀 Setting up Real-Time Lipsync Avatar System..."
echo "Hardware: 16GB GPU VRAM, 54GB RAM, 250GB+ Storage"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  This setup is optimized for Linux with NVIDIA GPU"
    echo "Current OS: $OSTYPE"
fi

# Check NVIDIA GPU
echo "🔍 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "❌ NVIDIA GPU not detected. Please install NVIDIA drivers."
    exit 1
fi

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first
echo "🔥 Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project structure..."
mkdir -p models/{sadtalker,musetalk,wav2lip}
mkdir -p cache/{audio,video,processed}
mkdir -p uploads/{images,audio}
mkdir -p outputs/{videos,previews}
mkdir -p logs
mkdir -p static/{css,js,uploads}
mkdir -p templates

# Download pre-trained models (this will be done in the Python scripts)
echo "🤖 Model downloads will be handled by the application..."

echo "✅ Setup complete! Your system is ready for real-time lipsync generation."
echo ""
echo "To start the application:"
echo "1. source venv/bin/activate"
echo "2. python main.py"
echo ""
echo "The web interface will be available at: http://localhost:8000"