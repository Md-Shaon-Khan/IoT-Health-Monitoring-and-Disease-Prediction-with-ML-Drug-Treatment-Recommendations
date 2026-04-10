# IoT HealthBridge - Disease Prediction & Medicine Recommendation System

একটি IoT-ভিত্তিক রোগ নির্ণয় এবং ঔষধ সুপারিশ সিস্টেম যা মেশিন লার্নিং ব্যবহার করে রোগীর স্বাস্থ্য ডেটা বিশ্লেষণ করে রোগ নির্ণয় এবং ঔষধ সুপারিশ প্রদান করে।

## Features

- **IoT Data Integration**: Temperature, Heart Rate, Blood Pressure, ECG monitoring
- **AI Disease Prediction**: 11-parameter ML model for disease prediction
- **Drug Recommendation**: AI-powered medicine suggestions
- **Real-time Dashboard**: Doctor and patient dashboards
- **MongoDB Atlas**: Cloud database for data storage
- **PWA Support**: Installable web app for mobile devices
- **Responsive Design**: Works on desktop, tablet, and mobile

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, JavaScript (PWA)
- **Database**: MongoDB Atlas
- **ML Models**: TensorFlow/Keras, Scikit-learn
- **Deployment**: Heroku/Railway/Render compatible

## Installation & Setup

### Prerequisites

- Python 3.11+
- MongoDB Atlas account
- Git

### Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/Md-Shaon-Khan/IoT-Health-Monitoring-and-Disease-Prediction-with-ML-Drug-Treatment-Recommendations.git
   cd IoT-Health-Monitoring-and-Disease-Prediction-with-ML-Drug-Treatment-Recommendations
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Setup MongoDB Atlas**
   - Create MongoDB Atlas cluster
   - Get connection string
   - Set environment variable: `MONGODB_URL=your_atlas_connection_string`

4. **Setup ML Models**
   - Download model files (contact developer for access)
   - Place in `model/` directory

5. **Run the application**
   ```bash
   uvicorn backend.app:app --reload
   ```

   - Open http://localhost:8000

## Deployment

### Free Hosting Options

#### Option 1: Heroku

1. Create Heroku account
2. Install Heroku CLI
3. ```bash
   heroku create your-app-name
   heroku config:set MONGODB_URL=your_atlas_connection_string
   heroku config:set GROQ_API_KEY=your_groq_api_key
   git push heroku pdf-only-clean:main
   ```

#### Option 2: Railway

1. Connect GitHub repository
2. Set environment variables:
   - `MONGODB_URL`
   - `GROQ_API_KEY`
3. Deploy automatically

#### Option 3: Render

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
4. Set environment variables

### Environment Variables

```bash
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/dbname
GROQ_API_KEY=your_groq_api_key_here
```

## Project Structure

```
├── backend/
│   ├── app.py              # FastAPI application
│   └── database.py         # MongoDB connection
├── frontend/
│   ├── index.html          # Landing page
│   ├── dashboard.html      # Main dashboard
│   ├── auth.html           # Login/Register
│   ├── manifest.json       # PWA manifest
│   ├── service-worker.js   # PWA service worker
│   └── *.css/*.js          # Styles and scripts
├── model/                  # ML model files (not in repo)
├── Database/               # Sample data
├── ECG/                    # ECG processing scripts
└── requirements.txt        # Python dependencies
```

## API Endpoints

- `POST /api/register` - User registration
- `POST /api/login` - User login
- `POST /api/predict` - Disease prediction
- `GET /api/doctor-stats` - Doctor dashboard stats
- `GET /api/doctor-patient-list` - Patient list for doctors
- `POST /api/send-feedback` - Send medical feedback

## Mobile App Installation

1. Open the deployed website on your phone
2. Tap "Add to Home Screen" in browser menu
3. The app will install as a PWA

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Create Pull Request

## License

This project is for educational purposes.

## Contact

For questions or model files, contact the developer.
