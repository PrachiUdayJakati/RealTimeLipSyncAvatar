# Real-Time Lipsync Avatar 🎭

A powerful AI-powered web application that generates high-quality lipsync videos from a single image and text input. Optimized for real-time performance on high-end GPU systems.

## ✨ Features

- **Real-time Video Generation**: Generate lipsync videos in seconds
- **Multiple AI Models**: Support for MuseTalk, SadTalker, and Wav2Lip
- **Advanced TTS Pipeline**: Multiple TTS providers with fallback options
- **GPU Optimized**: Designed for 16GB VRAM + 54GB RAM systems
- **Web Interface**: Beautiful, responsive web UI
- **API Support**: RESTful API for integration
- **Caching System**: Smart caching for faster repeated generations
- **Multiple Voice Options**: Support for various TTS voices and providers

## 🚀 Quick Start

### Prerequisites

- **Linux with NVIDIA GPU** (16GB VRAM recommended)
- **Python 3.8+**
- **CUDA 12.1+**
- **54GB+ RAM**
- **250GB+ Storage**

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RealTimeLipsyncAvatar
   ```

2. **Set up environment**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Configure API keys**
   ```bash
   cp .env.template .env
   # Edit .env and add your API keys
   nano .env
   ```

4. **Activate virtual environment**
   ```bash
   source venv/bin/activate
   ```

5. **Start the application**
   ```bash
   python main.py
   ```

6. **Access the web interface**
   Open your browser and go to: `http://localhost:8000`

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# ElevenLabs TTS API Key (required for high-quality TTS)
ELEVENLABS_API_KEY=your_api_key_here

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE_MB=100
CACHE_ENABLED=True
CACHE_SIZE_MB=500

# GPU Settings
CUDA_VISIBLE_DEVICES=0
MAX_VRAM_USAGE=0.8

# Web Server Settings
HOST=0.0.0.0
PORT=8000
WORKERS=1
```

### Supported AI Models

| Model | Quality | Speed | VRAM Usage |
|-------|---------|-------|------------|
| MuseTalk | High | Fast | 4.5GB |
| SadTalker | Very High | Medium | 6.0GB |
| Wav2Lip | Medium | Very Fast | 2.5GB |

### TTS Providers

- **ElevenLabs**: High-quality commercial TTS (requires API key)
- **Edge TTS**: Free Microsoft TTS service
- **ESPnet**: Open-source TTS models

## 📖 Usage

### Web Interface

1. **Upload Image**: Select a clear photo of a person's face
2. **Enter Text**: Type the text you want the person to say
3. **Choose Voice**: Select from available TTS voices
4. **Select Model**: Choose AI model (optional)
5. **Generate**: Click "Generate Video" and wait for processing
6. **Download**: Save the generated video

### API Usage

#### Generate Video
```bash
curl -X POST "http://localhost:8000/api/generate" \
  -F "image=@path/to/image.jpg" \
  -F "text=Hello, this is a test message" \
  -F "voice=Rachel" \
  -F "model_id=musetalk"
```

#### Generate Audio Only
```bash
curl -X POST "http://localhost:8000/api/generate-audio" \
  -F "text=Hello world" \
  -F "voice=Rachel" \
  --output audio.wav
```

#### Get System Status
```bash
curl "http://localhost:8000/api/status"
```

## 🏗️ Architecture

```
RealTimeLipsyncAvatar/
├── main.py                 # FastAPI web application
├── models/                 # AI model implementations
│   ├── model_manager.py    # Model management and GPU optimization
│   ├── musetalk_model.py   # MuseTalk implementation
│   └── __init__.py
├── utils/                  # Utility modules
│   ├── tts_pipeline.py     # Advanced TTS pipeline
│   ├── tts_eleven.py       # Legacy TTS (deprecated)
│   └── __init__.py
├── templates/              # HTML templates
│   └── index.html          # Main web interface
├── static/                 # Static assets
│   └── css/
│       └── style.css       # Custom styles
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── .env.template          # Environment template
└── README.md              # This file
```

## 🔧 Advanced Configuration

### GPU Memory Optimization

The application automatically manages GPU memory based on your hardware:

- **16GB VRAM**: Can run all models simultaneously
- **8GB VRAM**: Recommended to use one model at a time
- **4GB VRAM**: Use Wav2Lip model only

### Performance Tuning

1. **Adjust VRAM usage**:
   ```env
   MAX_VRAM_USAGE=0.8  # Use 80% of available VRAM
   ```

2. **Enable caching**:
   ```env
   CACHE_ENABLED=True
   CACHE_SIZE_MB=1000  # 1GB cache
   ```

3. **Optimize for speed**:
   - Use Wav2Lip model for fastest generation
   - Enable audio caching
   - Use Edge TTS for faster audio generation

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `MAX_VRAM_USAGE` in `.env`
   - Use a smaller model (Wav2Lip)
   - Restart the application

2. **TTS generation fails**
   - Check your `ELEVENLABS_API_KEY`
   - Verify internet connection
   - Try Edge TTS as fallback

3. **Slow video generation**
   - Ensure GPU drivers are up to date
   - Check GPU utilization with `nvidia-smi`
   - Reduce image resolution

4. **Web interface not accessible**
   - Check if port 8000 is available
   - Verify firewall settings
   - Try different port in `.env`

### Logs

Check application logs for detailed error information:
```bash
tail -f logs/app.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [MuseTalk](https://github.com/TMElyralab/MuseTalk) - Real-time high-quality lip synchronization
- [SadTalker](https://github.com/Winfredy/SadTalker) - High-quality talking head generation
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) - Fast lip synchronization
- [ElevenLabs](https://elevenlabs.io/) - High-quality text-to-speech
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**Note**: This application requires significant computational resources. Ensure your system meets the minimum requirements for optimal performance.